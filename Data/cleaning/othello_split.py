#!/usr/bin/env python3
"""
othello_split.py — label transcript segments as in-character (Othello text)
or out-of-character (present-day discussion) by fuzzy-matching against the play.

Subcommands
-----------
  label     Segment a transcript and auto-label every segment.
  sample    Draw a blind random sample for hand-labeling (validation set).
  evaluate  Compare your hand labels to the pipeline's labels (precision/recall/F1).

Typical workflow
----------------
  python othello_split.py label  --play othello.txt --script video1.srt --out video1_labeled.csv
  # open video1_labeled.csv, fill `final_label` for rows where auto_label == "review"
  python othello_split.py sample --labeled video1_labeled.csv --n 100 --out gold_sample.csv
  # fill `gold_label` in gold_sample.csv WITHOUT looking at the labeled file
  python othello_split.py evaluate --labeled video1_labeled.csv --gold gold_sample.csv

Labels
------
  in_character      segment matches a line (or run of lines) from the play
  out_of_character  no meaningful match: present-day discussion
  mixed             discussion that contains a quoted play line (see quoted_play_text)
  review            uncertain-band score; needs a human decision
  too_short         too few words to match reliably (e.g. "my lord", "yeah"); needs a human decision

Requires: pip install rapidfuzz
"""
from __future__ import annotations

import argparse
import csv
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    from rapidfuzz import fuzz
except ImportError:
    sys.exit("Missing dependency: run  pip install rapidfuzz")

LABELS = ["in_character", "out_of_character", "mixed"]  # 'mixed' only in --labels three

# --------------------------------------------------------------------------- #
# Text normalization
# --------------------------------------------------------------------------- #
_PUNCT = re.compile(r"[^a-z0-9\s]")
_SPACES = re.compile(r"\s+")


def normalize(text: str) -> str:
    text = text.lower().replace("’", "'").replace("‘", "'")
    text = text.replace("'", "")          # 'tis -> tis, it's -> its
    text = _PUNCT.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


STOPWORDS = set("""
a an and are as at be but by for from had has have he her him his i if in is it its
me my not of on or so that the their them then there they this to was we were what
when which who will with you your o oh yes no do did does all our us shall would
""".split())

ARCHAIC = set("""
thou thee thy thine ye hath doth dost art wilt shalt ere tis twas wherefore whence
hither thither nay aye prithee methinks mine anon forsooth alack alas oft perchance
whither yonder hast canst couldst wouldst shouldst didst mayst
""".split())


def content_tokens(norm: str) -> list[str]:
    return [t for t in norm.split() if t not in STOPWORDS and len(t) > 2]


def archaic_density(norm: str) -> float:
    toks = norm.split()
    if not toks:
        return 0.0
    hits = sum(1 for t in toks if t in ARCHAIC or (len(t) > 4 and t.endswith(("eth", "est"))))
    return round(hits / len(toks), 3)


# --------------------------------------------------------------------------- #
# Play loading
# --------------------------------------------------------------------------- #
_SPEAKER_PREFIX = re.compile(r"^\s*[A-Z][A-Z' ]{1,30}[.:]\s+")           # "OTHELLO. text" / "IAGO: text"
_STAGE = re.compile(r"\[.*?\]|\(.*?\)")
_SKIP_LINE = re.compile(r"^\s*(ACT|SCENE|Enter|Exit|Exeunt|Re-enter|Flourish|Dramatis)\b", re.I)


def load_play(path: Path) -> list[str]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    # Strip Project Gutenberg boilerplate if present
    m_start = re.search(r"\*\*\* ?START OF.*?\*\*\*", raw)
    m_end = re.search(r"\*\*\* ?END OF.*?\*\*\*", raw)
    if m_start:
        raw = raw[m_start.end(): m_end.start() if m_end else None]

    lines = []
    for line in raw.splitlines():
        s = line.strip()
        if not s or _SKIP_LINE.match(s):
            continue
        if s.isupper() and len(s.split()) <= 4:      # speaker heading like "OTHELLO"
            continue
        s = _STAGE.sub(" ", s)
        s = _SPEAKER_PREFIX.sub("", s)
        n = normalize(s)
        if len(n.split()) >= 2:
            lines.append(n)
    if len(lines) < 50:
        print(f"WARNING: only {len(lines)} usable lines parsed from the play file — check its format.",
              file=sys.stderr)
    return lines


def build_windows(play_lines: list[str], max_span: int = 3):
    """Windows of 1..max_span consecutive lines, since a spoken sentence can span verse lines."""
    windows = []
    for span in range(1, max_span + 1):
        for i in range(len(play_lines) - span + 1):
            windows.append(" ".join(play_lines[i:i + span]))
    index = defaultdict(set)
    for wid, w in enumerate(windows):
        for t in set(w.split()):
            index[t].add(wid)
    # rarer words count more when picking candidates (so "I am not what I am" still finds its line)
    n = len(windows)
    idf = {t: math.log((n + 1) / (len(ids) + 1)) + 0.1 for t, ids in index.items()}
    return windows, (index, idf)


# --------------------------------------------------------------------------- #
# Transcript loading + segmentation
# --------------------------------------------------------------------------- #
_TS = re.compile(r"(\d{1,2}:)?\d{1,2}:\d{2}[.,]\d{1,3}\s*-->\s*")
_SPEAKER_TAG = re.compile(r"^\s*([A-Za-z][\w .'-]{0,30}):\s+(.*)$")
_SENT_SPLIT = re.compile(r"(?<=[.!?;])\s+")


def load_transcript(path: Path) -> list[dict]:
    """Returns turns: {speaker, start, text}."""
    suffix = path.suffix.lower()
    turns = []
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                row = {k.lower().strip(): v for k, v in row.items()}
                if "text" not in row:
                    sys.exit("CSV transcript needs a 'text' column (optional: speaker, start).")
                turns.append({"speaker": row.get("speaker", ""), "start": row.get("start", ""),
                              "text": row["text"]})
        return turns

    raw = path.read_text(encoding="utf-8", errors="replace")
    if suffix in (".srt", ".vtt"):
        start = ""
        for line in raw.splitlines():
            s = line.strip()
            if not s or s == "WEBVTT" or s.isdigit():
                continue
            if _TS.search(s):
                start = s.split("-->")[0].strip()
                continue
            s = re.sub(r"<[^>]+>", "", s)               # strip vtt tags
            m = _SPEAKER_TAG.match(s)
            spk, txt = (m.group(1), m.group(2)) if m else ("", s)
            turns.append({"speaker": spk, "start": start, "text": txt})
        return turns

    # plain text: one turn per non-empty line, optional "Speaker: text"
    for line in raw.splitlines():
        s = line.strip()
        if not s:
            continue
        m = _SPEAKER_TAG.match(s)
        spk, txt = (m.group(1), m.group(2)) if m else ("", s)
        turns.append({"speaker": spk, "start": "", "text": txt})
    return turns


def segment(turns: list[dict], unit: str) -> list[dict]:
    segs = []
    for t in turns:
        pieces = [t["text"]] if unit in ("line", "turn") else _SENT_SPLIT.split(t["text"])
        for p in pieces:
            if p.strip():
                segs.append({"speaker": t["speaker"], "start": t["start"], "text": p.strip()})
    if unit == "turn":   # merge consecutive lines from the same speaker
        merged = []
        for s in segs:
            if merged and s["speaker"] and merged[-1]["speaker"] == s["speaker"]:
                merged[-1]["text"] += " " + s["text"]
            else:
                merged.append(dict(s))
        segs = merged
    return segs


# --------------------------------------------------------------------------- #
# Matching
# --------------------------------------------------------------------------- #
def best_match(norm: str, windows, index_idf, max_candidates: int = 300):
    index, idf = index_idf
    counts = Counter()
    for t in set(norm.split()):
        for wid in index.get(t, ()):
            counts[wid] += idf[t]
    if not counts:
        return 0.0, 0.0, None, ""
    cands = [wid for wid, _ in counts.most_common(max_candidates)]

    best_full, best_full_wid = 0.0, None
    best_part, best_part_wid, best_span = 0.0, None, ""
    for wid in cands:
        w = windows[wid]
        f = fuzz.ratio(norm, w)
        if f > best_full:
            best_full, best_full_wid = f, wid
        # partial match: a play line quoted inside a longer discussion sentence
        if len(w.split()) >= 4 and len(norm) > len(w) * 1.3:
            al = fuzz.partial_ratio_alignment(w, norm)
            if al.score > best_part:
                best_part, best_part_wid = al.score, wid
                best_span = norm[al.dest_start:al.dest_end]
    wid = best_full_wid if best_full >= best_part else best_part_wid
    return best_full, best_part, wid, best_span


def classify(full, part, n_words, hi, lo, min_words):
    if n_words < min_words:
        return "too_short"
    if full >= hi:
        return "in_character"
    if part >= hi:
        return "mixed"
    if max(full, part) < lo:
        return "out_of_character"
    return "review"


# --------------------------------------------------------------------------- #
# Subcommands
# --------------------------------------------------------------------------- #
def resolve_binary(rows, hi, lo):
    """Collapse to two labels. Uncertain rows are decided by their neighbours
    (in-character lines come in runs), then by score / archaic language."""
    first = []
    for r in rows:
        lab = r["auto_label"]
        if lab == "in_character":
            first.append("in_character")
        elif lab in ("out_of_character", "mixed"):      # a quote inside discussion is still discussion
            first.append("out_of_character")
        else:
            first.append(None)
    n = len(rows)
    for i, r in enumerate(rows):
        if first[i] is not None:
            r["final_label"], r["confidence"] = first[i], "high"
            continue
        prev = next((first[j] for j in range(i - 1, -1, -1) if first[j]), None)
        nxt = next((first[j] for j in range(i + 1, n) if first[j]), None)
        if prev and prev == nxt:
            r["final_label"], r["confidence"] = prev, "medium"
        else:
            score = max(r["full_score"], r["partial_score"])
            ic = score >= (hi + lo) / 2 or r["archaic_density"] >= 0.15
            r["final_label"] = "in_character" if ic else "out_of_character"
            r["confidence"] = "low"


def cmd_label(a):
    play = load_play(Path(a.play))
    windows, index = build_windows(play, a.max_span)
    segs = segment(load_transcript(Path(a.script)), a.unit)
    print(f"Play: {len(play)} lines -> {len(windows)} match windows | Transcript: {len(segs)} segments")

    rows = []
    for i, s in enumerate(segs, 1):
        norm = normalize(s["text"])
        full, part, wid, span = best_match(norm, windows, index)
        label = classify(full, part, len(norm.split()), a.hi, a.lo, a.min_words)
        rows.append({
            "segment_id": i, "speaker": s["speaker"], "start": s["start"], "text": s["text"],
            "auto_label": label,
            "final_label": label if label in LABELS else "",
            "confidence": "high" if label in LABELS else "",
            "full_score": round(full, 1), "partial_score": round(part, 1),
            "archaic_density": archaic_density(norm),
            "quoted_span": span if label == "mixed" else "",
            "matched_play_text": windows[wid] if wid is not None and max(full, part) >= a.lo else "",
        })
    if a.labels == "binary":
        resolve_binary(rows, a.hi, a.lo)

    fields = ["segment_id", "speaker", "start", "text", "final_label", "confidence", "auto_label",
              "full_score", "partial_score", "archaic_density", "quoted_span", "matched_play_text"]
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print("Final label counts:", dict(Counter(r["final_label"] or "UNRESOLVED" for r in rows)))
    if a.labels == "binary":
        print("Confidence:", dict(Counter(r["confidence"] for r in rows)),
              "-> spot-check the 'low' rows if you have time.")
        stem = Path(a.out).with_suffix("")
        for lab in ("in_character", "out_of_character"):
            out = Path(f"{stem}_{lab}.txt")
            out.write_text("\n".join(r["text"] for r in rows if r["final_label"] == lab) + "\n",
                           encoding="utf-8")
            print(f"Wrote {out}")
    else:
        todo = sum(1 for r in rows if not r["final_label"])
        print(f"{todo} rows need a human decision in `final_label`.")
    print(f"Wrote {a.out}")


def cmd_sample(a):
    with open(a.labeled, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    random.seed(a.seed)
    pick = random.sample(rows, min(a.n, len(rows)))
    pick.sort(key=lambda r: int(r["segment_id"]))
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["segment_id", "speaker", "text", "gold_label"])
        w.writeheader()
        for r in pick:   # blind: predicted labels and scores are NOT included
            w.writerow({"segment_id": r["segment_id"], "speaker": r["speaker"],
                        "text": r["text"], "gold_label": ""})
    print(f"Wrote {len(pick)} blind rows to {a.out} (seed={a.seed}). "
          f"Fill gold_label with one of: {', '.join(LABELS)}")


def cmd_evaluate(a):
    with open(a.labeled, newline="", encoding="utf-8") as f:
        pred = {r["segment_id"]: (r["final_label"] or r["auto_label"]) for r in csv.DictReader(f)}
    with open(a.gold, newline="", encoding="utf-8") as f:
        gold = {r["segment_id"]: r["gold_label"].strip() for r in csv.DictReader(f) if r["gold_label"].strip()}
    pairs = [(gold[k], pred.get(k, "missing")) for k in gold]
    if not pairs:
        sys.exit("No gold labels filled in yet.")

    classes = sorted(set(g for g, _ in pairs) | set(p for _, p in pairs))
    cm = Counter(pairs)
    acc = sum(cm[(c, c)] for c in classes) / len(pairs)
    print(f"\nEvaluated {len(pairs)} segments | accuracy = {acc:.3f}\n")
    print(f"{'class':<18}{'precision':>10}{'recall':>10}{'f1':>8}{'support':>9}")
    for c in classes:
        tp = cm[(c, c)]
        fp = sum(cm[(g, c)] for g in classes if g != c)
        fn = sum(cm[(c, p)] for p in classes if p != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        print(f"{c:<18}{prec:>10.3f}{rec:>10.3f}{f1:>8.3f}{tp + fn:>9}")
    print("\nConfusion matrix (rows = gold, cols = predicted):")
    print(" " * 18 + "".join(f"{c[:12]:>14}" for c in classes))
    for g in classes:
        print(f"{g:<18}" + "".join(f"{cm[(g, p)]:>14}" for p in classes))
    unresolved = [p for _, p in pairs if p in ("review", "too_short")]
    if unresolved:
        print(f"\nNote: {len(unresolved)} sampled rows still have no final_label — resolve them first.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    l = sub.add_parser("label", help="auto-label a transcript")
    l.add_argument("--play", required=True, help="plain-text Othello (Folger or Gutenberg .txt)")
    l.add_argument("--script", required=True, help="transcript: .txt, .srt, .vtt, or .csv")
    l.add_argument("--out", default="labeled.csv")
    l.add_argument("--labels", choices=["binary", "three"], default="binary",
                   help="binary: in_character / out_of_character only (default); three: also 'mixed' + manual review")
    l.add_argument("--unit", choices=["sentence", "line", "turn"], default="sentence")
    l.add_argument("--hi", type=float, default=85, help="score >= hi -> confident match")
    l.add_argument("--lo", type=float, default=60, help="score < lo -> confident non-match")
    l.add_argument("--min-words", type=int, default=4, help="shorter segments -> too_short")
    l.add_argument("--max-span", type=int, default=3, help="max consecutive play lines per window")
    l.set_defaults(func=cmd_label)

    s = sub.add_parser("sample", help="draw a blind validation sample")
    s.add_argument("--labeled", required=True)
    s.add_argument("--n", type=int, default=100)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--out", default="gold_sample.csv")
    s.set_defaults(func=cmd_sample)

    e = sub.add_parser("evaluate", help="precision/recall against hand labels")
    e.add_argument("--labeled", required=True)
    e.add_argument("--gold", required=True)
    e.set_defaults(func=cmd_evaluate)

    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
