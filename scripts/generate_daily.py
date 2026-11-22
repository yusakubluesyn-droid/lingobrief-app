import os
import json
import random
import re
from datetime import datetime, timezone, timedelta
import feedparser
import requests
from bs4 import BeautifulSoup
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# 1. 4分野の信頼できる大手メディアRSSフィード（全文取得が可能なもの）
FEEDS = {
    "Military & Geopolitics": [
        ("Al Jazeera", "https://www.aljazeera.com/xml/rss/all.xml"),
        ("BBC World", "http://feeds.bbci.co.uk/news/world/rss.xml"),
        ("France 24", "https://www.france24.com/en/rss")
    ],
    "Technology": [
        ("BBC Technology", "http://feeds.bbci.co.uk/news/technology/rss.xml"),
        ("The Guardian Tech", "https://www.theguardian.com/uk/technology/rss")
    ],
    "Environment": [
        ("The Guardian Environment", "https://www.theguardian.com/uk/environment/rss"),
        ("BBC Science & Environment", "http://feeds.bbci.co.uk/news/science_and_environment/rss.xml")
    ],
    "Economy": [
        ("BBC Business", "http://feeds.bbci.co.uk/news/business/rss.xml"),
        ("The Guardian Business", "https://www.theguardian.com/uk/business/rss")
    ]
}

# Gemini出力用の厳密なデータ構造定義
class VocabItem(BaseModel):
    en: str
    fr: str
    pos: str
    lemma: str
    cognate: str
    ja: str
    note: str

class ParagraphItem(BaseModel):
    en_tagged: str = Field(description="Original English paragraph with <w id='w1'>word</w> tags inserted ONLY around key vocabulary. Do NOT change a single letter or punctuation mark of the original text.")
    fr_b2_tagged: str = Field(description="Natural journalistic French translation (B2 level) with matching <w id='w1'>mot</w> tags.")
    fr_a2_tagged: str = Field(description="Simplified beginner French version (A2 level) with matching <w id='w1'>mot</w> tags.")

class ExpressionItem(BaseModel):
    en_phrase: str
    academic_alt: str
    fr_phrase: str
    cognate_hint: str
    ja_explanation: str

class SlashItem(BaseModel):
    fr_original: str
    en_meaning: str
    ja_meaning: str
    chunks: list[dict] # [{"role": "S", "text": "...", "note": "..."}]
    verbs: list[dict]  # [{"word": "...", "lemma": "...", "tense": "...", "meaning": "..."}]

class TravelItem(BaseModel):
    from_news_fr: str
    from_news_ja: str
    situation: str
    travel_fr: str
    travel_ja: str
    tip: str

class DailyLesson(BaseModel):
    title_en: str
    title_fr: str
    vocab: dict[str, VocabItem]
    paragraphs: list[ParagraphItem]
    expressions: list[ExpressionItem]
    slash_reading: list[SlashItem]
    travel_apps: list[TravelItem]
    quiz: dict

def fetch_verbatim_article():
    category = random.choice(list(FEEDS.keys()))
    source_name, rss_url = random.choice(FEEDS[category])
    feed = feedparser.parse(rss_url)
    
    for entry in feed.entries[:5]:
        url = entry.link
        try:
            res = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            soup = BeautifulSoup(res.text, "html.parser")
            # 本文の段落を抽出（短すぎる広告やキャプションを除外）
            paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
            valid_p = [p for p in paragraphs if len(p) > 80 and "cookie" not in p.lower() and "subscribe" not in p.lower()]
            
            # 学習に最適なボリューム（最初の4〜5段落、約250〜400語）を抽出
            if len(valid_p) >= 4:
                selected_p = valid_p[:5]
                return {
                    "category": category,
                    "source": source_name,
                    "url": url,
                    "title": entry.title,
                    "paragraphs": selected_p
                }
        except Exception as e:
            continue
    raise RuntimeError("記事の取得に失敗しました。")

def main():
    article = fetch_verbatim_article()
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    prompt = f"""
You are an expert English & French linguistics tutor for a Japanese graduate-level student (English: Eiken Pre-1 aiming for academic/native mastery; French: university beginner aiming for travel reading).

CRITICAL RULE FOR ENGLISH TEXT:
Below are the exact verbatim paragraphs from {article['source']}.
In `en_tagged`, you MUST preserve EVERY SINGLE CHARACTER, SPACE, AND PUNCTUATION MARK of the original English paragraphs identically. ONLY insert `<w id="w1">...</w>`, `<w id="w2">...</w>` tags around 18-24 important vocabulary words/phrases across the article.
If `re.sub(r'<[^>]+>', '', en_tagged)` does not match the original text, the system will fail.

Original Title: {article['title']}
Original Paragraphs:
{json.dumps(article['paragraphs'], ensure_ascii=False, indent=2)}

Tasks:
1. Tag 18-24 key English words/idioms with `<w id="w1">` through `<w id="w24">` without altering the original text.
2. Provide a natural B2 journalistic French translation (`fr_b2_tagged`) and a simplified A2 beginner French version (`fr_a2_tagged`), placing the EXACT SAME `<w id="wX">` tags around the corresponding French words so clicking one highlights the other.
3. Populate `vocab` dictionary for every `w1`..`wN` with Japanese translation, POS, French lemma, and English-French cognate hints.
4. Provide 4 detailed expression breakdowns (`expressions`) comparing academic English alternatives and French cognates, explained in natural Japanese.
5. Provide 2 French grammar slash-reading breakdowns (`slash_reading`) from the B2 text (splitting S, V, O, M and explaining verb conjugations/lemmas in Japanese).
6. Provide 2 travel/daily French conversion examples (`travel_apps`) reusing vocabulary from the news article in realistic tourist situations (cafes, trains, hotels).
7. Create a 3-question `quiz` object matching this exact schema:
   - `q1`: {{"question": "English fill-in-the-blank sentence using a key idiom from the text (use _____)", "options": ["opt1", "opt2", "opt3", "opt4"], "answer": 0, "explanation": "Japanese explanation"}}
   - `q2`: {{"question": "French vocabulary meaning question in Japanese", "options": ["opt1", "opt2", "opt3", "opt4"], "answer": 1, "explanation": "Japanese explanation"}}
   - `q3`: {{"prompt": "Japanese prompt asking the user to write 1 English sentence on the article's topic", "model_answer": "Sophisticated academic/native English model sentence", "notes": "Japanese explanation of key expressions used"}}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=DailyLesson,
            temperature=0.2
        )
    )

    lesson_data = json.loads(response.text)

    # 原文完全一致の自動検証とフェイルセーフ
    for i, orig_p in enumerate(article["paragraphs"]):
        tagged_p = lesson_data["paragraphs"][i]["en_tagged"]
        stripped = re.sub(r"</?w[^>]*>", "", tagged_p)
        if stripped.strip() != orig_p.strip():
            # 万が一AIが1文字でも変えていた場合、原文を強制復元
            lesson_data["paragraphs"][i]["en_tagged"] = orig_p

    jst = timezone(timedelta(hours=9))
    today_str = datetime.now(jst).strftime("%Y-%m-%d")

    output = {
        "date": today_str,
        "category": article["category"],
        "source": article["source"],
        "url": article["url"],
        "lesson": lesson_data
    }

    os.makedirs("public/data/archive", exist_ok=True)
    with open("public/data/latest.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    with open(f"public/data/archive/{today_str}.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Successfully generated daily lesson for {today_str}: {article['title']}")

if __name__ == "__main__":
    main()
