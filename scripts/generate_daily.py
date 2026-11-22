import os
import json
import random
import re
import time
from datetime import datetime, timezone, timedelta
import feedparser
import requests
from bs4 import BeautifulSoup
from google import genai
from google.genai import types

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

def fetch_verbatim_article():
    categories = list(FEEDS.keys())
    random.shuffle(categories)
    
    for category in categories:
        sources = list(FEEDS[category])
        random.shuffle(sources)
        for source_name, rss_url in sources:
            try:
                feed = feedparser.parse(rss_url)
                for entry in feed.entries[:6]:
                    url = entry.link
                    res = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
                    soup = BeautifulSoup(res.text, "html.parser")
                    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
                    valid_p = [
                        p for p in paragraphs 
                        if len(p) > 80 
                        and "cookie" not in p.lower() 
                        and "subscribe" not in p.lower()
                        and "sign up" not in p.lower()
                    ]
                    if len(valid_p) >= 4:
                        return {
                            "category": category,
                            "source": source_name,
                            "url": url,
                            "title": entry.title,
                            "paragraphs": valid_p[:5]
                        }
            except Exception:
                continue
    raise RuntimeError("記事の取得に失敗しました。")

def generate_with_fallback(client, prompt):
    # 実在する正規モデルで粘り強くリトライする
    models_to_try = [
        "gemini-3.8-flash",
        "gemini-3.8-flash-lite",
        "gemini-3.8-flash",
        "gemini-3.8-flash-lite",
        "gemini-3.8-flash"
    ]
    last_error = None
    for idx, model_name in enumerate(models_to_try):
        try:
            print(f"Attempt {idx + 1}: Trying {model_name}...")
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.2
                )
            )
            return response.text
        except Exception as e:
            last_error = e
            print(f"{model_name} failed ({e}). Waiting 25s before retry...")
            time.sleep(25)
    raise last_error

def main():
    article = fetch_verbatim_article()
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    prompt = f"""
You are an expert English & French linguistics tutor for a Japanese graduate-level student (English: Eiken Pre-1 aiming for academic/native mastery; French: university beginner aiming for travel reading).

CRITICAL RULE FOR ENGLISH TEXT:
Below are the exact verbatim paragraphs from {article['source']}.
In `en_tagged`, you MUST preserve EVERY SINGLE CHARACTER, SPACE, AND PUNCTUATION MARK of the original English paragraphs identically. ONLY insert `<w id="w1">...</w>`, `<w id="w2">...</w>` tags around 18-24 important vocabulary words/phrases across the article.

Original Title: {article['title']}
Original Paragraphs:
{json.dumps(article['paragraphs'], ensure_ascii=False, indent=2)}

Return ONLY a valid JSON object matching this exact structure (no markdown fences):
{{
  "title_en": "Original English title",
  "title_fr": "French translation of title",
  "vocab": {{
    "w1": {{
      "en": "english word/phrase",
      "fr": "french word/phrase",
      "pos": "Part of speech in Japanese (e.g. 名詞, 動詞)",
      "lemma": "French dictionary form",
      "cognate": "Cognate hint in Japanese (e.g. 英語と同語源 / 固有語)",
      "ja": "Japanese translation",
      "note": "Usage nuance or grammar tip in Japanese"
    }}
  }},
  "paragraphs": [
    {{
      "en_tagged": "Verbatim paragraph 1 with <w id='w1'>...</w> tags",
      "fr_b2_tagged": "Journalistic B2 French translation with matching <w id='w1'>...</w> tags",
      "fr_a2_tagged": "Simplified A2 beginner French version with matching <w id='w1'>...</w> tags"
    }}
  ],
  "expressions": [
    {{
      "en_phrase": "Key English phrase from text",
      "academic_alt": "More formal/academic alternative expression",
      "fr_phrase": "Corresponding French phrase",
      "cognate_hint": "English-French connection hint in Japanese",
      "ja_explanation": "Detailed nuance and usage explanation in Japanese"
    }}
  ],
  "slash_reading": [
    {{
      "fr_original": "Sentence from B2 French text",
      "en_meaning": "English meaning",
      "ja_meaning": "Japanese meaning",
      "chunks": [
        {{"role": "S", "text": "Subject chunk", "note": "Grammar note in Japanese"}},
        {{"role": "V", "text": "Verb chunk", "note": "Grammar note in Japanese"}},
        {{"role": "O", "text": "Object/Complement chunk", "note": "Grammar note in Japanese"}},
        {{"role": "M", "text": "Modifier chunk", "note": "Grammar note in Japanese"}}
      ],
      "verbs": [
        {{"word": "conjugated verb", "lemma": "infinitive", "tense": "tense name in Japanese", "meaning": "Japanese meaning"}}
      ]
    }}
  ],
  "travel_apps": [
    {{
      "from_news_fr": "Phrase from article",
      "from_news_ja": "Japanese meaning",
      "situation": "Tourist situation (e.g. パリのカフェで)",
      "travel_fr": "Practical travel French sentence reusing the word",
      "travel_ja": "Japanese translation",
      "tip": "Practical tip in Japanese"
    }}
  ],
  "quiz": {{
    "q1": {{
      "question": "English fill-in-the-blank sentence using a key idiom from the text (use _____)",
      "options": ["option 1", "option 2", "option 3", "option 4"],
      "answer": 0,
      "explanation": "Explanation in Japanese"
    }},
    "q2": {{
      "question": "French vocabulary meaning question in Japanese",
      "options": ["option 1", "option 2", "option 3", "option 4"],
      "answer": 1,
      "explanation": "Explanation in Japanese"
    }},
    "q3": {{
      "prompt": "Prompt in Japanese asking to write 1 English sentence on the topic",
      "model_answer": "Sophisticated academic English model sentence",
      "notes": "Explanation of key expressions in Japanese"
    }}
  }}
}}

Ensure `expressions` has 4 items, `slash_reading` has 2 items, and `travel_apps` has 2 items.
"""

    raw_json_text = generate_with_fallback(client, prompt)
    lesson_data = json.loads(raw_json_text)

    # 原文完全一致の自動検証とフェイルセーフ
    for i, orig_p in enumerate(article["paragraphs"]):
        if i < len(lesson_data["paragraphs"]):
            tagged_p = lesson_data["paragraphs"][i]["en_tagged"]
            stripped = re.sub(r"</?w[^>]*>", "", tagged_p)
            if stripped.strip() != orig_p.strip():
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
