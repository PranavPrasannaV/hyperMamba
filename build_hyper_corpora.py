#!/usr/bin/env python3
"""
build_hyper_corpora.py
Generates three files:
 - corpus_nl.txt   (natural language)
 - corpus_code.txt (code snippets)
 - corpus_mixed.txt (tweets / emoji / URLs / math)
Target lines per file configurable.
"""

import os
import re
import random
from pathlib import Path
from datasets import load_dataset

TARGET_LINES = 20000   # change if you want more/fewer
OUT_DIR = Path(".")
random.seed(42)

# Simple cleaning helpers
def clean_text(s: str) -> str:
    s = s.strip()
    # normalize whitespace
    s = re.sub(r'\s+', ' ', s)
    # remove control chars
    s = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', s)
    return s

def sample_and_write(records, path: Path, target):
    seen = set()
    out = []
    for r in records:
        t = clean_text(r)
        if not t:
            continue
        if len(t) < 3:
            continue
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
        if len(out) >= target:
            break
    random.shuffle(out)
    path.write_text("\n".join(out), encoding="utf-8")
    print(f"Wrote {len(out)} lines to {path}")

def main(target=TARGET_LINES):
    # 1) Natural language: use wikipedia or bookcorpus (wiki is big; datasets returns samples)
    print("Loading natural language samples (wikipedia:20220301.en)...")
    try:
        ds_wiki = load_dataset("wikipedia", "20220301.en", split="train")
        # sample by taking random indices to avoid long pages
        records = []
        for i in random.sample(range(len(ds_wiki)), min(50000, len(ds_wiki))):
            text = ds_wiki[i]["text"]
            # take short paragraphs / first line
            if not text:
                continue
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            if lines:
                # pick a short line
                for ln in lines:
                    if 30 < len(ln) < 400:
                        records.append(ln)
                        break
            if len(records) >= target * 3:
                break
        sample_and_write(records, OUT_DIR / "corpus_nl.txt", target)
    except Exception as e:
        print("Failed to load wikipedia via datasets:", e)
        # fallback: use 'bookcorpus' or 'wikitext' if wikipedia fails
        try:
            ds = load_dataset("wikitext", "wikitext-103-v1", split="train")
            records = [l for l in ds["text"] if l and len(l) > 40]
            sample_and_write(records, OUT_DIR / "corpus_nl.txt", target)
        except Exception as e2:
            print("Fallback also failed:", e2)

              # 2) Code corpus: use codeparrot-clean (small, permissive) instead of old code_search_net
    print("Loading code samples (jtatman/python-code-dataset-500k)...")
    try:
        ds_code = load_dataset("jtatman/python-code-dataset-500k", split="train")
        records = []
        for ex in ds_code.shuffle(seed=42).select(range(min(200000, len(ds_code)))):
            snippet = ex.get("content") or ex.get("code") or ex.get("text")
            if not snippet:
                continue
            if 10 < len(snippet) < 800:
                records.append(snippet)
            if len(records) >= target * 2:
                break
        sample_and_write(records, OUT_DIR / "corpus_code.txt", target)
    except Exception as e:
        print("Failed to load codeparrot-clean:", e)







    # 3) Mixed: tweets (tweet_eval) + arXiv math subset (raw)
    print("Loading mixed samples (tweet_eval + arxiv sample)...")
    mixed_records = []
    try:
        ds_tweet = load_dataset("tweet_eval", "emoji", split="train")
        for ex in ds_tweet.shuffle(seed=42).select(range(min(50000, len(ds_tweet)))):
            txt = ex.get("text") or ex.get("tweet") or ex.get("content")
            if not txt:
                continue
            if len(txt) > 3 and len(txt) < 300:
                mixed_records.append(txt)
            if len(mixed_records) >= target:
                break
    except Exception as e:
        print("tweet_eval load failed:", e)

    # add some arXiv math LaTeX samples if available
    try:
        ds_arxiv = load_dataset("arxiv_dataset", split="train")
        for ex in ds_arxiv.shuffle(seed=43).select(range(20000)):
            title = ex.get("title","")
            abstract = ex.get("abstract","")
            text = (title + " " + abstract).strip()
            if text and len(text) < 500:
                # keep ones with math symbols
                if any(sym in text for sym in ["\\int", "\\sum", "\\frac", "π", "√", "^"]):
                    mixed_records.append(text)
            if len(mixed_records) >= target * 1.2:
                break
    except Exception:
        pass

    # if not enough, fallback to mixing NL and code examples
    if len(mixed_records) < target:
        try:
            ds_small = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
            extra = [l for l in ds_small["text"] if l and len(l) < 200][:target]
            mixed_records.extend(extra)
        except Exception:
            pass

    sample_and_write(mixed_records, OUT_DIR / "corpus_mixed.txt", target)

if __name__ == "__main__":
    main()
