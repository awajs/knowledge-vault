#!/usr/bin/env python3
from pathlib import Path
import argparse, json, os, sys, requests
sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrieval import search  # noqa

BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
SYSTEM = (
    "You are KnowledgeVault's filings analyst.\n"
    "Answer ONLY using the EVIDENCE provided.\n"
    "If the evidence is insufficient, answer exactly: \"Not enough evidence to answer.\"\n"
    "Return STRICT JSON with keys: answer (string), citations (array of {doc_id,source,page,quote})."
)

def make_user_prompt(question, hits):
    lines = [f"Question:\n{question}\n\nEVIDENCE SNIPPETS:"]
    for i, h in enumerate(hits, start=1):
        content = h.get("text") or h.get("snippet") or ""
        if len(content) > 1400: content = content[:1400] + " ..."
        lines.append(f"\n[{i}] doc_id={h.get('doc_id')} source={h.get('source')} page=null\n{content}")
    lines.append("\nReturn ONLY the JSON.")
    return "\n".join(lines)

def call_ollama(model, system, user):
    payload = {"model": model, "system": system, "prompt": user, "format": "json",
               "options": {"temperature": 0.2, "num_predict": 700}, "stream": False}
    r = requests.post(f"{BASE_URL}/api/generate", json=payload, timeout=600)
    r.raise_for_status()
    return r.json().get("response", "")

def main():
    ap = argparse.ArgumentParser(description="Ask Q&A over index (JSON answers + citations)")
    ap.add_argument("question")
    ap.add_argument("--index-dir", default="plugins/knowledgevault/data/index")
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--model", default="llama3.2:3b-instruct-q4_0")
    args = ap.parse_args()
    hits = search(args.index_dir, args.question, top_k=args.k)
    user = make_user_prompt(args.question, hits)
    try:
        data = json.loads(call_ollama(args.model, SYSTEM, user))
    except Exception as e:
        data = {"answer": f"Ollama error: {e}", "citations": []}
    if "citations" not in data:
        data["citations"] = [{"doc_id": h["doc_id"], "source": h["source"], "page": None,
                              "quote": (h.get("snippet") or "")[:180]} for h in hits[:3]]
    print(json.dumps(data, indent=2))

if __name__ == "__main__":
    main()
