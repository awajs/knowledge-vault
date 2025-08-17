#!/usr/bin/env python3
from pathlib import Path
import argparse, sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrieval import build_index  # noqa

def main():
    ap = argparse.ArgumentParser(description="Build KnowledgeVault index")
    ap.add_argument("roots", nargs="+", help="Directories to index (txt files)")
    ap.add_argument("--index-dir", default="plugins/knowledgevault/data/index")
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--trees", type=int, default=50)
    args = ap.parse_args()
    build_index(args.index_dir, *args.roots, model_name=args.model, trees=args.trees)

if __name__ == "__main__":
    main()
