#!/usr/bin/env python3
from pathlib import Path
import argparse, json, os, sys, hashlib, re, requests
sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrieval import search  # noqa

BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b-instruct-q4_0")

QUERIES = [
  "Item 7 Management's Discussion and Analysis",
  "Item 8 Financial Statements and Supplementary Data",
  "Consolidated Statements of Operations",
  "Net revenue Gross profit Operating income Net income",
  "Revenue growth gross margin operating margin cash from operations",
  "Transaction volume TPV GMV take rate",
]

SYSTEM = (
  "You are KnowledgeVault's Metrics Extractor.\n"
  "Use ONLY the EVIDENCE snippets provided. Do not invent numbers.\n"
  "Extract metrics if present: Revenue, Net revenue, Gross profit, Operating income/loss, Net income/loss, Gross margin %, Operating margin %, Cash from operations, TPV/GMV (and periods).\n"
  "Return STRICT JSON:\n"
  "{\n"
  '  "answer": "short summary (<=2 sentences) or \\"Not enough evidence to answer.\\"",\n'
  '  "metrics": [\n'
  '    {"name":"string","value":"string","unit":"string","period":"string|null","doc_id":"string","source":"string","page":null,"quote":"string"}\n'
  "  ]\n"
  "}\n"
  "Only include metrics clearly present in the evidence. If none, use metrics: []."
)

def _num_score(txt: str) -> int:
    if not txt: return 0
    m = len(re.findall(r'[$€£]\\s?\\d', txt))
    p = len(re.findall(r'\\b\\d{1,3}%\\b', txt))
    k = len(re.findall(r'\\b(?:revenue|net revenue|gross profit|operating income|net income|tpv|gmv|cash from operations)\\b', txt, flags=re.I))
    return 3*m + 2*p + k

def gather_hits(index_dir: str, company_or_ticker: str | None, k_per_query: int = 6, max_hits: int = 32):
    bag = {}
    company = company_or_ticker or ""
    queries = ([f"{company} {q}" for q in QUERIES] + ([company] if company else []))
    for q in queries:
        for h in search(index_dir, q, top_k=k_per_query):
            key = hashlib.md5((h.get('source','') + '|' + h.get('text', h.get('snippet',''))).encode('utf-8')).hexdigest()
            if key not in bag: bag[key] = h
            if len(bag) >= max_hits: break
        if len(bag) >= max_hits: break
    hits = list(bag.values())
    hits.sort(key=lambda h: _num_score(h.get('text', h.get('snippet','')) or ""), reverse=True)
    return hits

def make_user_prompt(question: str, hits: list[dict]) -> str:
    lines = []
    for i, h in enumerate(hits, start=1):
        content = h.get('text', h.get('snippet','')) or ""
        if len(content) > 1400: content = content[:1400] + " ..."
        lines.append(f"[{i}] doc_id={h.get('doc_id')} source={h.get('source')} page=null\\n{content}")
    return f"Question:\\n{question}\\n\\nEVIDENCE SNIPPETS:\\n" + "\\n\\n".join(lines) + "\\n\\nReturn ONLY the JSON."

def call_ollama(model: str, system: str, user: str, temperature: float = 0.15, max_tokens: int = 700) -> str:
    payload = {"model": model, "system": system, "prompt": user, "format": "json",
               "options": {"temperature": temperature, "num_predict": max_tokens}, "stream": False}
    r = requests.post(f"{BASE_URL}/api/generate", json=payload, timeout=600)
    if r.status_code != 200:
        try: err = r.json().get("error", r.text)
        except Exception: err = r.text
        raise RuntimeError(f"Ollama API error {r.status_code}: {err}")
    return r.json().get("response", "")

def main():
    ap = argparse.ArgumentParser(description="Extract filings metrics with JSON + citations")
    ap.add_argument("--company", default="", help="Company name or ticker for query bias")
    ap.add_argument("--index-dir", default="plugins/knowledgevault/data/index")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--k-per-query", type=int, default=6)
    ap.add_argument("--max-hits", type=int, default=32)
    args = ap.parse_args()

    hits = gather_hits(args.index_dir, args.company or None, k_per_query=args.k_per_query, max_hits=args.max_hits)
    user = make_user_prompt("Extract key financial metrics (values + units + periods).", hits)
    raw = call_ollama(args.model, SYSTEM, user)
    try:
        data = json.loads(raw)
    except Exception:
        data = {"answer": "Not enough evidence to answer.", "metrics": []}
    if not isinstance(data, dict) or "metrics" not in data or "answer" not in data:
        data = {"answer": "Not enough evidence to answer.", "metrics": []}
    for m in data.get("metrics", []):
        m.setdefault("page", None); m.setdefault("quote", "")
    print(json.dumps(data, indent=2))

if __name__ == "__main__":
    main()
