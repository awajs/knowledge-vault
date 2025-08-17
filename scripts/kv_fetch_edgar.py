#!/usr/bin/env python3
import time, json, re, os
from pathlib import Path
from typing import List, Optional
import requests
from bs4 import BeautifulSoup
import html2text

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "edgar" / "raw"
TXT_DIR = ROOT / "data" / "edgar" / "txt"
CACHE_DIR = ROOT / "data" / "edgar" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

SEC_USER_AGENT = os.environ.get("SEC_USER_AGENT", "KnowledgeVault/0.1 (your.email@example.com)")
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": SEC_USER_AGENT, "Accept-Encoding": "gzip, deflate"})

def _sleep(sec=0.5):
    time.sleep(sec)

def get_company_tickers():
    cache = CACHE_DIR / "company_tickers.json"
    if cache.exists():
        return json.loads(cache.read_text())
    url = "https://www.sec.gov/files/company_tickers.json"
    r = SESSION.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()
    cache.write_text(json.dumps(data))
    return data

def ticker_to_cik(ticker: str) -> Optional[str]:
    t = ticker.strip().upper()
    data = get_company_tickers()
    for _, rec in data.items():
        if rec.get("ticker", "").upper() == t:
            return f"{int(rec['cik_str']):010d}"
    return None

def get_recent_submissions(cik: str) -> dict:
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    r = SESSION.get(url, timeout=30)
    r.raise_for_status()
    return r.json()

def filings_for(cik: str, forms: List[str], limit: int = 10):
    sub = get_recent_submissions(cik)
    recent = sub.get("filings", {}).get("recent", {})
    out = []
    for form, acc, doc in zip(recent.get("form", []),
                              recent.get("accessionNumber", []),
                              recent.get("primaryDocument", [])):
        if form in forms:
            out.append((form, acc, doc))
        if len(out) >= limit:
            break
    return out

def filing_url(cik: str, accession: str, primary_doc: str) -> str:
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-','')}/{primary_doc}"

def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script", "style"]): t.extract()
    md = html2text.HTML2Text()
    md.ignore_images = True
    md.ignore_links = True
    md.body_width = 0
    txt = md.handle(str(soup))
    import re
    txt = re.sub(r"[ \t]+\n", "\n", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt.strip()

def download_and_convert(ticker: str, forms: List[str], limit: int = 5):
    cik = ticker_to_cik(ticker)
    if not cik:
        print(f"[MISS] Could not resolve CIK for {ticker}")
        return 0
    got = 0
    for form, acc, doc in filings_for(cik, forms, limit=limit):
        url = filing_url(cik, acc, doc)
        raw_dir = RAW_DIR / ticker.upper() / acc.replace("-", "")
        txt_dir = TXT_DIR / ticker.upper()
        raw_dir.mkdir(parents=True, exist_ok=True)
        txt_dir.mkdir(parents=True, exist_ok=True)

        raw_path = raw_dir / doc
        txt_path = txt_dir / f"{acc.replace('-','')}_{form}.txt"
        if txt_path.exists():
            print(f"[SKIP] {txt_path} exists")
            continue

        try:
            r = SESSION.get(url, timeout=60)
            if r.status_code == 429:
                retry = int(r.headers.get("Retry-After", "1"))
                print(f"[RATE] 429, sleeping {retry}s...")
                time.sleep(retry)
                r = SESSION.get(url, timeout=60)
            r.raise_for_status()
            raw_path.write_bytes(r.content)
            text = html_to_text(r.text)
            txt_path.write_text(text, encoding="utf-8")
            print(f"[OK] {ticker} {form} {acc} → {txt_path}")
            got += 1
        except Exception as e:
            print(f"[ERR] {ticker} {form} {acc}: {e}")
        _sleep(0.5)
    return got

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Fetch EDGAR filings → TXT")
    ap.add_argument("--tickers", nargs="+", required=True, help="e.g., MQ AAPL MSFT")
    ap.add_argument("--forms", nargs="+", default=["10-K","10-Q","S-1"])
    ap.add_argument("--limit", type=int, default=5)
    args = ap.parse_args()

    print(f"Using SEC User-Agent: {SEC_USER_AGENT}")
    total = 0
    for t in args.tickers:
        total += download_and_convert(t, args.forms, limit=args.limit)
    print(f"\\nDone. New TXT filings: {total}")

if __name__ == "__main__":
    main()
