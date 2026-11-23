#!/usr/bin/env python3
"""
GoldSample.py: build ONE hand-labeling sheet from ALL the *_labeled.csv files that
OthelloSplit.py wrote, then score the split against your hand labels.

OthelloSplit.py's own `sample` / `evaluate` work on one transcript at a time. This
script does the same job across every session at once, so the accuracy you report
in the paper covers the whole corpus.

Step 1, make the sheet (predicted labels and scores are NOT copied into it):
    python GoldSample.py sample --labeled "path/to/labeled/*_labeled.csv" --n 200 --out gold_sample_all.csv

Step 2, open gold_sample_all.csv in Excel or Google Sheets and type
    in_character   or   out_of_character
in the gold_label column for every row (see LABELING_GUIDE.md). Save it as CSV.

Step 3, score it:
    python GoldSample.py evaluate --labeled "path/to/labeled/*_labeled.csv" --gold gold_sample_all.csv

Only needs the Python standard library.
"""
from __future__ import annotations

import argparse
import csv
import glob
import random
import sys
from collections import Counter
from pathlib import Path

LABELS = ("in_character", "out_of_character")
SHORTHAND = {"in": "in_character", "i": "in_character", "ic": "in_character",
             "out": "out_of_character", "o": "out_of_character", "ooc": "out_of_character"}


def labeled_files(pattern: str) -> list[Path]:
    files = sorted(Path(p) for p in glob.glob(pattern))
    if not files:
        sys.exit(f"No files matched {pattern!r}. Point --labeled at the *_labeled.csv files "
                 f"that `OthelloSplit.py label` wrote.")
    return files


def read_rows(path: Path) -> list[dict]:
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def cmd_sample(a):
    files = labeled_files(a.labeled)
    pool = []   # every segment in every session: (file name, row index, rows)
    per_file = {}
    for p in files:
        rows = read_rows(p)
        per_file[p.name] = rows
        pool.extend((p.name, i) for i in range(len(rows)))
    if not pool:
        sys.exit("The labeled files contain no rows.")

    random.seed(a.seed)
    pick = random.sample(pool, min(a.n, len(pool)))
    pick.sort(key=lambda t: (t[0], t[1]))

    out_fields = ["row", "file", "segment_id", "speaker", "previous_line", "text", "next_line", "gold_label"]
    with open(a.out, "w", newline="", encoding="utf-8-sig") as f:   # utf-8-sig opens cleanly in Excel
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()
        for n, (name, i) in enumerate(pick, 1):
            rows = per_file[name]
            r = rows[i]
            w.writerow({
                "row": n,
                "file": name,
                "segment_id": r["segment_id"],
                "speaker": r.get("speaker", ""),
                "previous_line": rows[i - 1]["text"] if i > 0 else "",
                "text": r["text"],
                "next_line": rows[i + 1]["text"] if i + 1 < len(rows) else "",
                "gold_label": "",
            })
    sessions = len({name for name, _ in pick})
    print(f"Wrote {len(pick)} rows from {sessions} of {len(files)} sessions to {a.out} (seed={a.seed}).")
    print("Fill gold_label with in_character or out_of_character (in / out also work). "
          "Do not open the *_labeled.csv files while you label.")


def norm_label(s: str) -> str:
    s = s.strip().lower().replace("-", "_").replace(" ", "_")
    return SHORTHAND.get(s, s)


def cmd_evaluate(a):
    preds = {}
    for p in labeled_files(a.labeled):
        for r in read_rows(p):
            preds[(p.name, r["segment_id"])] = r.get("final_label") or r.get("auto_label", "")

    with open(a.gold, newline="", encoding="utf-8-sig") as f:
        gold_rows = [r for r in csv.DictReader(f) if r.get("gold_label", "").strip()]
    if not gold_rows:
        sys.exit("No gold labels filled in yet.")

    pairs, bad, skipped = [], [], 0
    for r in gold_rows:
        g = norm_label(r["gold_label"])
        if g == "skip":
            skipped += 1
            continue
        if g not in LABELS:
            bad.append((r.get("row"), r["gold_label"]))
            continue
        pred = preds.get((r["file"], r["segment_id"]), "missing")
        pred = "out_of_character" if pred == "mixed" else pred   # a quote inside talk counts as talk
        pairs.append((g, pred))
    if bad:
        print("Skipped rows with an unrecognized gold_label:", bad[:10], "..." if len(bad) > 10 else "")

    if skipped:
        print(f"Left out {skipped} rows you marked skip (report this number in the paper).")
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
    print("\nConfusion matrix (rows = your label, columns = the script's label):")
    print(" " * 18 + "".join(f"{c[:16]:>18}" for c in classes))
    for g in classes:
        print(f"{g:<18}" + "".join(f"{cm[(g, p)]:>18}" for p in classes))
    left = sum(1 for _, p in pairs if p not in LABELS)
    if left:
        print(f"\nNote: {left} sampled rows have no final label in the labeled files "
              f"(review / too_short / missing). Count them as errors or resolve them first.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample", help="write a blind labeling sheet drawn from all sessions")
    s.add_argument("--labeled", required=True, help='glob for the labeled CSVs, in quotes, e.g. "labeled/*_labeled.csv"')
    s.add_argument("--n", type=int, default=200)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--out", default="gold_sample_all.csv")
    s.set_defaults(func=cmd_sample)
    e = sub.add_parser("evaluate", help="score the split against your hand labels")
    e.add_argument("--labeled", required=True)
    e.add_argument("--gold", required=True)
    e.set_defaults(func=cmd_evaluate)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
