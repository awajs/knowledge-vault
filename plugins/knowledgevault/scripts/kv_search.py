#!/usr/bin/env python3
from pathlib import Path
import argparse, json, sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrieval import search  # noqa

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--index-dir", default="plugins/knowledgevault/data/index")
    ap.add_argument("--k", type=int, default=8)
    args = ap.parse_args()
    hits = search(args.index_dir, args.query, top_k=args.k)
    print(json.dumps(hits, indent=2))

if __name__ == "__main__":
    main()
