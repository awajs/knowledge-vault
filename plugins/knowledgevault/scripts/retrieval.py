from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Iterable
import json, re
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer
from annoy import AnnoyIndex

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"  # 384-dim, small & fast

@dataclass
class Segment:
    doc_id: str
    source: str
    text: str
    idx: int

def _read_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""

def _iter_txt_files(roots: List[Path]) -> Iterable[Path]:
    for root in roots:
        if not root.exists(): 
            continue
        for p in root.rglob("*.txt"):
            if p.is_file():
                yield p

def _chunk(text: str, target_chars: int = 2200, overlap_chars: int = 300) -> List[str]:
    text = re.sub(r"\s+\n", "\n", text).strip()
    if not text:
        return []
    out = []
    i = 0
    while i < len(text):
        j = min(len(text), i + target_chars)
        out.append(text[i:j])
        if j == len(text): break
        i = max(0, j - overlap_chars)
    return out

def build_index(index_dir: str, *roots: str, model_name: str = DEFAULT_MODEL, trees: int = 50) -> Dict:
    index_dir = Path(index_dir); index_dir.mkdir(parents=True, exist_ok=True)
    ann_path = index_dir / "annoy.ann"
    meta_path = index_dir / "meta.jsonl"
    info_path = index_dir / "info.json"

    segs: List[Segment] = []
    files = list(_iter_txt_files([Path(r) for r in roots]))
    for p in tqdm(files, desc="Indexing"):
        text = _read_text(p)
        if not text.strip(): continue
        doc_id = p.stem
        src = str(p)
        for idx, ch in enumerate(_chunk(text)):
            segs.append(Segment(doc_id=doc_id, source=src, text=ch, idx=idx))

    model = SentenceTransformer(model_name)
    dim = model.get_sentence_embedding_dimension()
    u = AnnoyIndex(dim, "angular")
    with meta_path.open("w", encoding="utf-8") as mf:
        for i, s in enumerate(segs):
            v = model.encode(s.text, normalize_embeddings=True).astype(np.float32)
            u.add_item(i, v)
            mf.write(json.dumps(asdict(s)) + "\n")
    u.build(trees)
    u.save(str(ann_path))
    info = {"model_name": model_name, "dim": dim, "trees": trees, "n_segments": len(segs)}
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))
    return info

def _load_meta(meta_path: Path) -> List[Dict]:
    out = []
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            try: out.append(json.loads(line))
            except Exception: pass
    return out

def search(index_dir: str, query: str, top_k: int = 8, model_name: str = DEFAULT_MODEL) -> List[Dict]:
    index_dir = Path(index_dir)
    ann_path = index_dir / "annoy.ann"
    meta_path = index_dir / "meta.jsonl"
    info_path = index_dir / "info.json"
    if not (ann_path.exists() and meta_path.exists()):
        raise FileNotFoundError(f"Index not found at {index_dir}")

    info = json.loads(info_path.read_text(encoding="utf-8"))
    model = SentenceTransformer(info.get("model_name", model_name))
    dim = info.get("dim", model.get_sentence_embedding_dimension())
    u = AnnoyIndex(dim, "angular"); u.load(str(ann_path))
    meta = _load_meta(meta_path)

    qv = model.encode([query], normalize_embeddings=True)[0].astype(np.float32)
    ids, dists = u.get_nns_by_vector(qv, top_k, include_distances=True)
    out = []
    for rank, (i, d) in enumerate(zip(ids, dists), start=1):
        s = meta[i]; text = s.get("text", "")
        snip = (re.sub(r"\s+", " ", text).strip())[:240]
        out.append({"rank": rank, "score": float(1.0 - d/2.0),
                    "doc_id": s.get("doc_id"), "source": s.get("source"),
                    "text": text, "snippet": snip})
    return out
