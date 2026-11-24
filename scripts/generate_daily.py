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
                    container = soup.find("article") or soup.find("main") or soup
                    paragraphs = [p.get_text(" ", strip=True) for p in container.find_all("p")]
                    valid_p = [
                        p for p in paragraphs 
                        if len(p) > 65 
                        and not any(ng in p.lower() for ng in ["cookie", "subscribe", "sign up", "newsletter", "all rights reserved", "follow us", "read more:"])
                    ]
                    if len(valid_p) >= 4:
                        return {
                            "category": category,
                            "source": source_name,
                            "url": url,
                            "title": entry.title,
                            "paragraphs": valid_p[:15]
                        }
            except Exception:
                continue
    raise RuntimeError("記事の取得に失敗しました。")

def generate_with_fallback(client, prompt):
    # 混雑している3.8系が503の場合、別サーバー群の2.5系へ確実にフォールバックさせる
    models_to_try = [
        "gemini-3.8-flash",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-3.8-flash-lite",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite"
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
            print(f"Success with {model_name}!")
            return response.text
        except Exception as e:
            last_error = e
            wait_sec = 15 if idx < 2 else 30
            print(f"{model_name} failed ({e}). Waiting {wait_sec}s before fallback...")
            time.sleep(wait_sec)
    raise last_error

def translate_chunk(client, source, paragraphs_chunk, start_word_idx):
    prompt = f"""
You are an expert English & French linguistics tutor for a Japanese graduate student.
CRITICAL RULE: In `en_tagged`, preserve EVERY SINGLE CHARACTER, SPACE, AND PUNCTUATION MARK of the original English paragraphs identically. ONLY insert `<w id="w{start_word_idx}">...</w>`, `<w id="w{start_word_idx+1}">...</w>` tags around 10-14 important vocabulary words/phrases in this section.

Paragraphs:
{json.dumps(paragraphs_chunk, ensure_ascii=False, indent=2)}

Return ONLY a valid JSON object:
{{
  "vocab": {{
    "w{start_word_idx}": {{
      "en": "english word/phrase",
      "fr": "french word/phrase",
      "pos": "Part of speech in Japanese",
      "lemma": "French dictionary form",
      "cognate": "Cognate hint in Japanese",
      "ja": "Japanese translation",
      "note": "Usage nuance or grammar tip in Japanese"
    }}
  }},
  "paragraphs": [
    {{
      "en_tagged": "Verbatim paragraph with <w id='w...'>...</w> tags",
      "fr_b2_tagged": "Journalistic B2 French translation with matching <w id='w...'>...</w> tags",
      "fr_a2_tagged": "Simplified A2 beginner French version with matching <w id='w...'>...</w> tags"
    }}
  ]
}}
"""
    raw = generate_with_fallback(client, prompt)
    return json.loads(raw)

def main():
    article = fetch_verbatim_article()
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    all_p = article["paragraphs"]
    mid = (len(all_p) + 1) // 2
    chunk1_p = all_p[:mid]
    chunk2_p = all_p[mid:]

    res1 = translate_chunk(client, article["source"], chunk1_p, 1)
    merged_vocab = dict(res1.get("vocab", {}))
    merged_paragraphs = list(res1.get("paragraphs", []))

    if chunk2_p:
        res2 = translate_chunk(client, article["source"], chunk2_p, 30)
        merged_vocab.update(res2.get("vocab", {}))
        merged_paragraphs.extend(res2.get("paragraphs", []))

    for i, orig_p in enumerate(all_p):
        if i < len(merged_paragraphs):
            tagged_p = merged_paragraphs[i]["en_tagged"]
            stripped = re.sub(r"</?w[^>]*>", "", tagged_p)
            if stripped.strip() != orig_p.strip():
                merged_paragraphs[i]["en_tagged"] = orig_p

    study_prompt = f"""
You are an expert English & French linguistics tutor for a Japanese graduate student (English: Eiken Pre-1 aiming for academic mastery; French: beginner aiming for travel reading).

Title: {article['title']}
Full Text:
{chr(10).join(all_p)}

Return ONLY a valid JSON object:
{{
  "title_en": {json.dumps(article['title'])},
  "title_fr": "Natural French translation of the title",
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
      "fr_original": "French sentence based on the article",
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
      "from_news_fr": "French phrase from article",
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
    study_data = json.loads(generate_with_fallback(client, study_prompt))

    lesson_data = {
        "title_en": article["title"],
        "title_fr": study_data["title_fr"],
        "vocab": merged_vocab,
        "paragraphs": merged_paragraphs,
        "expressions": study_data["expressions"],
        "slash_reading": study_data["slash_reading"],
        "travel_apps": study_data["travel_apps"],
        "quiz": study_data["quiz"]
    }

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

    index_path = "public/data/index.json"
    archive_list = []
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            archive_list = json.load(f)
    archive_list = [item for item in archive_list if item.get("date") != today_str]
    archive_list.insert(0, {
        "date": today_str,
        "category": article["category"],
        "source": article["source"],
        "title_en": article["title"]
    })
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(archive_list[:90], f, ensure_ascii=False, indent=2)

    print(f"Successfully generated full lesson ({len(all_p)} paragraphs) for {today_str}: {article['title']}")

if __name__ == "__main__":
    main()
