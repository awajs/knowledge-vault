#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, re, sys, urllib.request, urllib.error
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------

ALIASES = {
    "NVDA": {"NVDA", "NVIDIA", "NVIDIA CORP", "NVIDIA CORPORATION"},
    "AAPL": {"AAPL", "APPLE", "APPLE INC", "APPLE INC."},
    "MSFT": {"MSFT", "MICROSOFT", "MICROSOFT CORP", "MICROSOFT CORPORATION"},
}

GENERIC_INCOME_HEADINGS = [
    r"condensed\s+consolidated\s+statements?\s+of\s+income",
    r"consolidated\s+statements?\s+of\s+income",
    r"condensed\s+consolidated\s+statements?\s+of\s+operations",
    r"consolidated\s+statements?\s+of\s+operations",
    r"income\s+statement",
    r"consolidated\s+statements?\s+of\s+income\s+and\s+comprehensive\s+income",
]

# Require a close-by heading that looks like the income statement, with units/EPS nearby
STRICT_HEADING = re.compile(r"(consolidated\s+statements?\s+of\s+(income|operations)[^\n]{0,200})", re.I | re.DOTALL)
STRICT_NEARBY_UNITS_OR_EPS = re.compile(
    r"(?:\((?:in\s+millions|in\s+thousands)[^)]*\)|except\s+per\s+share|earnings\s+per\s+share)",
    re.I,
)

# Row patterns
PFX = r"^[^\w]{0,6}"
ROW_ALIASES = {
    "netRevenue": [PFX + r"net\s+revenue", PFX + r"total\s+revenue", PFX + r"revenue\b", PFX + r"sales\b"],
    "costOfRevenue": [PFX + r"cost\s+of\s+revenue", PFX + r"cost\s+of\s+goods\s+sold"],
    "operatingIncome": [PFX + r"operating\s+income", PFX + r"income\s+from\s+operations", PFX + r"operating\s+loss"],
    "netIncome": [PFX + r"net\s+income", PFX + r"net\s+loss", PFX + r"net\s+earnings"],
    "basicEPS": [PFX + r"(earnings|net\s+income)\s+per\s+share.*basic", PFX + r"basic(\s+earnings)?\s+per\s+share"],
    "dilutedEPS": [PFX + r"(earnings|net\s+income)\s+per\s+share.*diluted", PFX + r"diluted(\s+earnings)?\s+per\s+share"],
}

MAX_CONTEXT_CHARS = 32000
WINDOW_BEFORE = 300
WINDOW_AFTER = 50000

# --------------------------------------------------------------------------------------
# Utils
# --------------------------------------------------------------------------------------

def expand_aliases(s: str) -> List[str]:
    key = s.strip().upper()
    return sorted(list(ALIASES.get(key, {key})))

def safe_read(path: Path, max_bytes: int = 8_000_000) -> str:
    try:
        return path.read_text("utf-8", errors="ignore")[:max_bytes]
    except Exception:
        return ""

def file_is_company(fp: Path, raw: str, aliases: List[str], primary: str) -> bool:
    in_path = f"/{primary}/" in str(fp).upper().replace("\\", "/")
    mentions = any(a.lower() in raw.lower() for a in aliases)
    return in_path or mentions

def likely_table(text: str) -> bool:
    t = text.lower()
    return (
        "(in millions" in t
        or "in millions" in t
        or "(in thousands" in t
        or "in thousands" in t
        or "except per share" in t
        or text.count("  ") > 10
        or "|" in text
    )

def unit_scale(text: str) -> int:
    t = text.lower()
    if "in millions" in t:
        return 1_000_000
    if "in thousands" in t:
        return 1_000
    return 1

def coerce_float(v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    neg = s.startswith("(") and s.endswith(")")
    if neg:
        s = s[1:-1]
    s = s.replace(",", "").replace("$", "")
    try:
        val = float(s)
        return -val if neg else val
    except Exception:
        return None

# --------------------------------------------------------------------------------------
# Window finding
# --------------------------------------------------------------------------------------

def _find_strict_income_windows(raw: str) -> List[Dict[str, int]]:
    wins = []
    low = raw.lower()
    for m in STRICT_HEADING.finditer(low):
        anchor = m.start()
        fwd = raw[anchor : min(len(raw), anchor + 1200)]
        if STRICT_NEARBY_UNITS_OR_EPS.search(fwd):
            wins.append(
                {
                    "start": max(0, anchor - WINDOW_BEFORE),
                    "end": min(len(raw), anchor + WINDOW_AFTER),
                    "anchor": anchor,
                    "heading": raw[m.start() : m.end()],
                }
            )
    return wins

def _find_generic_income_windows(raw: str) -> List[Dict[str, int]]:
    wins = []
    low = raw.lower()
    for pat in GENERIC_INCOME_HEADINGS:
        for m in re.finditer(pat, low, flags=re.I):
            anchor = m.start()
            wins.append(
                {
                    "start": max(0, anchor - WINDOW_BEFORE),
                    "end": min(len(raw), anchor + WINDOW_AFTER),
                    "anchor": anchor,
                    "heading": raw[m.start() : m.end()],
                }
            )
    if not wins:
        return []
    wins.sort(key=lambda w: (w["start"], w["end"]))
    merged = [wins[0]]
    for w in wins[1:]:
        a = merged[-1]
        if w["start"] <= a["end"] + 200:
            a["end"] = max(a["end"], w["end"])
        else:
            merged.append(w)
    return merged

def has_units(s: str) -> bool:
    t = s.lower()
    return "in millions" in t or "in thousands" in t or "except per share" in t

def has_eps(s: str) -> bool:
    t = s.lower()
    return "earnings per share" in t or "per share—basic" in t or "per share - basic" in t

def _has_key_rowlines(s: str) -> bool:
    rev_line = re.search(r"(?m)^\W*(net\s+revenue|total\s+revenue|revenue|sales)\b", s, re.I) is not None
    net_line = re.search(r"(?m)^\W*net\s+(income|loss|earnings)\b", s, re.I) is not None
    return rev_line and net_line

def is_percent_table(snippet: str) -> bool:
    t = snippet.lower()
    if "expressed as a percentage of revenue" in t:
        if has_units(t) or has_eps(t):
            return False
        return True
    head = snippet[:2000]
    perc = head.count("%")
    dollars = head.count("$")
    if (perc >= 5) and not (has_units(head) or has_eps(head)) and dollars == 0:
        return True
    return False

def _looks_like_stock_comp_note(heading: str, fwd: str) -> bool:
    """
    Detect 'Consolidated Statements of Income include stock-based compensation ... as follows'
    and similar variants. Drop these windows unconditionally because they are not the primary
    income statement; they're breakdown tables.
    """
    h = (heading or "").lower()
    f = (fwd or "")[:600].lower()
    has_sbc = ("stock-based compensation" in h) or ("stock-based compensation" in f)
    has_marker = ("as follows" in h) or ("as follows" in f) or ("include" in h) or ("include" in f)
    if not (has_sbc and has_marker):
        return False
    # If it actually has proper rowlines, don't drop as note.
    if _has_key_rowlines(fwd):
        return False
    return True

def build_income_context_from_files(files: List[Path], aliases: List[str], primary: str, debug: bool):
    contexts, citations = [], []
    for fp in files:
        raw = safe_read(fp)
        if not raw or not file_is_company(fp, raw, aliases, primary):
            continue

        wins = _find_strict_income_windows(raw) or _find_generic_income_windows(raw)
        if debug:
            print(f"[DEBUG] {fp.name}: income-statement windows found = {len(wins)}", file=sys.stderr)

        for w in wins:
            start, end, anchor = w["start"], w["end"], w["anchor"]
            fwd = raw[anchor:end]
            snippet = raw[start:end]
            heading = w.get("heading", "")

            # EARLY DROP: stock-based compensation include/as-follows breakdowns
            if _looks_like_stock_comp_note(heading, fwd):
                if debug:
                    print(f"[DEBUG] anchor@{anchor} heading='{heading}' decision=DROP reason=stock-comp-include", file=sys.stderr)
                continue

            # DROP percent-of-revenue MD&A tables
            if is_percent_table(fwd):
                if debug:
                    print(f"[DEBUG] anchor@{anchor} heading='{heading}' decision=DROP reason=percent-table", file=sys.stderr)
                continue

            # Require real row starts
            if not _has_key_rowlines(fwd):
                if debug:
                    print(f"[DEBUG] anchor@{anchor} heading='{heading}' decision=DROP reason=no-rowlines", file=sys.stderr)
                continue

            # Must look table-ish
            if not likely_table(fwd):
                if debug:
                    print(f"[DEBUG] anchor@{anchor} heading='{heading}' decision=DROP reason=not-tableish", file=sys.stderr)
                continue

            # Strong signal for primary table
            force_keep = has_units(fwd) or has_eps(fwd)

            if debug:
                peek = re.sub(r"\s+", " ", fwd[:480])[:240]
                print(
                    f"[DEBUG] anchor@{anchor} heading='{heading}' decision=KEEP reason={'force-keep' if force_keep else 'ok'} | peek: '{peek}...'",
                    file=sys.stderr,
                )

            contexts.append(f"[Source: {fp}]\n{snippet}")
            citations.append({"source": str(fp), "snippet": snippet[:240].replace("\n", " ")})
    return contexts, citations

# --------------------------------------------------------------------------------------
# Deterministic parser
# --------------------------------------------------------------------------------------

def _split_numeric_columns(line: str) -> List[str]:
    line = line.replace("|", " ")
    line = re.sub(r"\t", " ", line)
    return [p.strip() for p in re.split(r"\s{2,}", line.strip()) if p.strip()]

def _find_numeric_trail(parts: List[str]) -> List[str]:
    nums = []
    for p in reversed(parts):
        if re.search(r"[0-9]", p):
            nums.append(p)
        elif nums:
            break
    return list(reversed(nums))

def parse_income_table_block(block: str, latest_on: str = "left") -> Dict[str, Any]:
    lines = [re.sub(r"[^\S\r\n]+", " ", ln).strip() for ln in block.splitlines() if ln.strip()]
    if not lines:
        return {}
    comp = {k: [re.compile(p, re.I) for p in pats] for k, pats in ROW_ALIASES.items()}
    metrics: Dict[str, Any] = {}
    for ln in lines:
        low = ln.lower()
        row = None
        for k, regs in comp.items():
            if any(r.match(low) for r in regs):
                row = k
                break
        if not row:
            continue
        parts = _split_numeric_columns(ln)
        if len(parts) < 2:
            continue
        tail = _find_numeric_trail(parts)
        if len(tail) < 2:
            continue
        token = tail[0] if latest_on == "left" else tail[-1]
        val = coerce_float(token)
        if val is not None:
            metrics[row] = val
    return metrics

def parse_income_windows_det(contexts: List[str], prefer: str = "left") -> Optional[Dict[str, Any]]:
    for orient in [prefer, "right" if prefer == "left" else "left"]:
        for ctx in contexts:
            block = re.sub(r"^\[Source:.*?\]\n", "", ctx, flags=re.DOTALL).strip()
            parsed = parse_income_table_block(block, latest_on=orient)
            if parsed.get("netRevenue") is not None and parsed.get("netIncome") is not None:
                parsed["_orientation"] = orient
                return parsed
    return None

# --------------------------------------------------------------------------------------
# Fallback chunking
# --------------------------------------------------------------------------------------

def chunk_text(text: str, maxlen=1500) -> List[str]:
    parts = []
    for block in re.split(r"\n\s*\n", text):
        b = block.strip()
        if not b:
            continue
        if len(b) <= maxlen:
            parts.append(b)
        else:
            for i in range(0, len(b), maxlen):
                parts.append(b[i : i + maxlen])
    return parts

def score_chunk(text: str, aliases: List[str]) -> int:
    t = text.lower()
    tags = [
        "consolidated statements of income",
        "consolidated statement of income",
        "statements of operations",
        "statement of operations",
        "income statement",
        "net revenue",
        "revenue",
        "sales",
        "cost of revenue",
        "cost of goods sold",
        "operating income",
        "operating loss",
        "net income",
        "net loss",
        "earnings per share",
        "basic",
        "diluted",
        "(in millions",
        "in thousands",
    ]
    score = sum(tag in t for tag in tags)
    if any(a.lower() in t for a in aliases):
        score += 1
    if likely_table(text):
        score += 2
    if (
        "consolidated statements of income" in t
        or "statements of operations" in t
        or "income statement" in t
    ):
        score += 3
    return score

def prefer_10k(filename: str) -> int:
    fn = filename.lower()
    if "_10-k" in fn or re.search(r"\b10[-\s]?k\b", fn):
        return 3
    if "_10-q" in fn or re.search(r"\b10[-\s]?q\b", fn):
        return 1
    return 0

def pick_top_chunks(index_dir: Path, company: str, k: int, top: int, debug: bool) -> List[Dict[str, Any]]:
    aliases = expand_aliases(company)
    files = sorted([p for p in index_dir.rglob("*.txt") if p.is_file()])
    if debug:
        print(f"[DEBUG] Scanning {len(files)} files under {index_dir}", file=sys.stderr)
    primary = company.strip().upper()
    cands: List[Tuple[int, str, str]] = []
    for fp in files:
        raw = safe_read(fp)
        if not raw or not file_is_company(fp, raw, aliases, primary):
            continue
        fbonus = prefer_10k(str(fp))
        for ch in chunk_text(raw, maxlen=1500):
            if not any(a.lower() in ch.lower() for a in aliases):
                continue
            s = score_chunk(ch, aliases) + fbonus
            if s > 0:
                cands.append((s, str(fp), ch))
    if not cands:
        return []
    cands.sort(key=lambda x: x[0], reverse=True)
    cands = cands[: max(k, top)]

    def rerank_key(item):
        s, src, ch = item
        t = ch.lower()
        bonus = 0
        if "consolidated statements of income" in t or "statements of operations" in t or "income statement" in t:
            bonus += 5
        if likely_table(ch):
            bonus += 3
        if re.search(r"\((in|In)\s+(millions|thousands)\)", ch):
            bonus += 2
        return s + bonus

    cands.sort(key=rerank_key, reverse=True)
    final = cands[:top]
    if debug:
        print(f"[DEBUG] Picked {len(final)} fallback chunks (strict {primary} filter):", file=sys.stderr)
        for i, (s, src, ch) in enumerate(final):
            first = " ".join(ch.split()[:20])
            print(f"  - #{i+1} score={s} file={src} snippet='{first}...'", file=sys.stderr)
    return [{"source": src, "text": ch} for _, src, ch in final]

# --------------------------------------------------------------------------------------
# Ollama
# --------------------------------------------------------------------------------------

def ollama_pull(model: str, timeout: float = 900.0) -> None:
    data = json.dumps({"name": model, "stream": True}).encode("utf-8")
    req = urllib.request.Request(
        "http://127.0.0.1:11434/api/pull",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    urllib.request.urlopen(req, timeout=timeout).read()

def ollama_generate(model: str, prompt: str, timeout: float = 240.0) -> str:
    options = {"num_ctx": 8192, "temperature": 0}
    payload = json.dumps({"model": model, "prompt": prompt, "stream": True, "options": options}).encode("utf-8")

    def _call():
        req = urllib.request.Request(
            "http://127.0.0.1:11434/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        chunks = []
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for line in resp:
                try:
                    obj = json.loads(line.decode("utf-8"))
                    if "response" in obj:
                        chunks.append(obj["response"])
                except Exception:
                    continue
        return "".join(chunks).strip()

    try:
        return _call()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            # pull then retry once
            ollama_pull(model)
            return _call()
        raise

# --------------------------------------------------------------------------------------
# LLM JSON helpers
# --------------------------------------------------------------------------------------

def make_extraction_prompt(context: str) -> str:
    return (
        "Extract ONLY the latest FULL FISCAL YEAR from the Consolidated Statements of Income/Operations. "
        "Respect units ('in millions'/'in thousands'); parentheses mean negative. Return ONLY this JSON:\n"
        "{\n"
        '  "period": {"startDate": null, "endDate": null},\n'
        '  "incomeStatement": {\n'
        '    "netRevenue": null,\n'
        '    "costOfRevenue": null,\n'
        '    "operatingIncome": null,\n'
        '    "netIncome": null,\n'
        '    "basicEPS": null,\n'
        '    "dilutedEPS": null\n'
        "  }\n"
        "}\n\nContext:\n\"\"\"\n"
        + context
        + "\n\"\"\"\nReturn JSON only, with plain numbers (no commas)."
    )

def strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s.replace("```", "")
    if s.endswith("```"):
        s = s.rsplit("```", 1)[0]
    return s.strip()

def _remove_thousands_commas_outside_strings(s: str) -> str:
    out = []
    in_str = False
    esc = False
    for ch in s:
        if in_str:
            out.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
                out.append(ch)
            else:
                out.append(ch)
    s2 = "".join(out)
    s2 = re.sub(r'(?<!")(?P<a>\d),(?P<b>\d{3})(?!")', r"\g<a>\g<b>", s2)
    return s2

def rescue_json_with_braces(s: str) -> Optional[str]:
    starts = [i for i, ch in enumerate(s) if ch == "{"]
    best = None
    best_len = -1
    for st in starts:
        d = 0
        for i in range(st, len(s)):
            c = s[i]
            if c == "{":
                d += 1
            elif c == "}":
                d -= 1
                if d == 0:
                    seg = s[st : i + 1]
                    if len(seg) > best_len:
                        best, best_len = seg, len(seg)
                    break
    return best

def parse_llm_json(s: str) -> Dict[str, Any]:
    s = strip_code_fences(s)
    s = _remove_thousands_commas_outside_strings(s)
    try:
        return json.loads(s)
    except Exception:
        seg = rescue_json_with_braces(s)
        if seg:
            seg = _remove_thousands_commas_outside_strings(seg)
            return json.loads(seg)
        raise ValueError("Failed to parse JSON from model output.")

# --------------------------------------------------------------------------------------
# Validation & output
# --------------------------------------------------------------------------------------

def is_reasonable(inc: Dict[str, Any]) -> bool:
    rev = coerce_float(inc.get("netRevenue"))
    cor = coerce_float(inc.get("costOfRevenue"))
    opi = coerce_float(inc.get("operatingIncome"))
    ni = coerce_float(inc.get("netIncome"))
    if rev is None or rev <= 0:
        return False
    if cor is not None and (cor < 0 or cor > 2.5 * rev):
        return False
    if opi is not None and abs(opi) > 2.0 * rev:
        return False
    if ni is not None and abs(ni) > 2.0 * rev:
        return False
    return True

def build_output(
    ok: bool,
    inc: Optional[Dict[str, Any]] = None,
    period: Optional[Dict[str, Any]] = None,
    citations: Optional[List[Dict[str, str]]] = None,
    reason: Optional[str] = None,
) -> Dict[str, Any]:
    if not ok:
        out = {"answer": "Not enough evidence to answer.", "metrics": []}
        if reason:
            out["reason"] = reason
        if citations:
            out["citations"] = citations
        return out
    metrics = []

    def add(name, val, eps=False):
        v = coerce_float(val)
        if v is not None:
            metrics.append({"name": name, "value": v, "unit": "USD/share" if eps else "USD"})

    add("netRevenue", inc.get("netRevenue"))
    add("costOfRevenue", inc.get("costOfRevenue"))
    add("operatingIncome", inc.get("operatingIncome"))
    add("netIncome", inc.get("netIncome"))
    add("basicEPS", inc.get("basicEPS"), True)
    add("dilutedEPS", inc.get("dilutedEPS"), True)

    out = {"answer": "OK" if metrics else "Not enough evidence to answer.", "metrics": metrics}
    if period:
        out["period"] = period
    if citations:
        out["citations"] = citations
    return out

# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--company", required=True)
    ap.add_argument("--index-dir", required=True)
    ap.add_argument("--model", default="mistral:7b-instruct")
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--no-sanity", action="store_true")
    args = ap.parse_args()

    index_dir = Path(args.index_dir)
    if not index_dir.exists():
        print(
            json.dumps(
                {"answer": "Not enough evidence to answer.", "metrics": [], "error": f"index-dir not found: {index_dir}"}
            )
        )
        return

    aliases = expand_aliases(args.company)
    primary = args.company.strip().upper()

    files = [p for p in sorted(index_dir.rglob("*.txt")) if p.is_file()]
    company_files = []
    for fp in files:
        raw = safe_read(fp)
        if raw and file_is_company(fp, raw, aliases, primary):
            company_files.append(fp)

    if args.debug:
        print(f"[DEBUG] Candidate files for {primary}: {len(company_files)}", file=sys.stderr)
        for fp in company_files:
            print(f"  - {fp}", file=sys.stderr)

    tenk = [fp for fp in company_files if re.search(r"\b10[-\s]?k\b", fp.name.lower())]
    files_for_windows = tenk or company_files

    contexts, citations = build_income_context_from_files(files_for_windows, aliases, primary, args.debug)

    # Deterministic pass (both orientations)
    if contexts:
        det = parse_income_windows_det(contexts, prefer="left")
        if det:
            scale = unit_scale("\n\n".join(contexts))
            inc = {
                k: (coerce_float(det.get(k)) * scale if det.get(k) is not None else None)
                for k in ("netRevenue", "costOfRevenue", "operatingIncome", "netIncome")
            }
            inc["basicEPS"] = det.get("basicEPS")
            inc["dilutedEPS"] = det.get("dilutedEPS")

            if not args.no_sanity and not is_reasonable(inc):
                if args.debug:
                    print("[DEBUG] Sanity failed with first orientation; retrying with opposite orientation.", file=sys.stderr)
                det2 = parse_income_windows_det(contexts, prefer="right")
                if det2:
                    inc = {
                        k: (coerce_float(det2.get(k)) * scale if det2.get(k) is not None else None)
                        for k in ("netRevenue", "costOfRevenue", "operatingIncome", "netIncome")
                    }
                    inc["basicEPS"] = det2.get("basicEPS")
                    inc["dilutedEPS"] = det2.get("dilutedEPS")
                    if args.no_sanity or is_reasonable(inc):
                        print(json.dumps(build_output(True, inc, None, citations), indent=2))
                        return
            else:
                print(json.dumps(build_output(True, inc, None, citations), indent=2))
                return

    # Fallback: scored chunks + LLM
    if not contexts and args.debug:
        print("[DEBUG] No income-statement windows kept; falling back to scored chunks", file=sys.stderr)
    if not contexts:
        chunks = pick_top_chunks(index_dir, args.company, k=args.k, top=args.top, debug=args.debug)
        if not chunks:
            print(
                json.dumps(
                    build_output(
                        False, reason="No candidate chunks matched income statement patterns or aliases"
                    ),
                    indent=2,
                )
            )
            return
        contexts = [f"[Source: {c['source']}]\n{c['text']}" for c in chunks]
        citations = [{"source": c["source"], "snippet": c["text"][:240].replace("\n", " ")} for c in chunks]

    context = ("\n\n" + ("-" * 80) + "\n\n").join(contexts)
    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS]

    prompt = make_extraction_prompt(context)
    try:
        resp = ollama_generate(args.model, prompt, timeout=240.0)
    except Exception as e:
        print(json.dumps(build_output(False, reason=f"Ollama error: {e}", citations=citations), indent=2))
        return

    if args.debug:
        print("[DEBUG] Raw model response (truncated to 800 chars):", file=sys.stderr)
        print(resp[:800], file=sys.stderr)

    try:
        data = parse_llm_json(resp)
    except Exception as e:
        reason = f"Failed to parse JSON from model output: {e}"
        if args.debug:
            reason += " (see raw model response above)"
        print(json.dumps(build_output(False, reason=reason, citations=citations), indent=2))
        return

    inc = (data or {}).get("incomeStatement", {}) or {}
    scale = unit_scale(context)
    for k in ("netRevenue", "costOfRevenue", "operatingIncome", "netIncome"):
        if inc.get(k) is not None:
            v = coerce_float(inc[k])
            inc[k] = None if v is None else v * scale

    if not args.no_sanity and not is_reasonable(inc):
        print(json.dumps(build_output(False, reason="Sanity check failed (LLM)", citations=citations), indent=2))
        return

    print(json.dumps(build_output(True, inc, (data or {}).get("period"), citations), indent=2))

if __name__ == "__main__":
    main()
