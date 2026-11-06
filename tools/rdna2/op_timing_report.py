#!/usr/bin/env python3
"""GPU time per custom op on prefill steps, from VLLM_RDNA_OP_TIMING csvs. Prompts are split at host gaps > 1.5 s and grouped by size.
self = op duration minus its nested ops. other = gaps between top-level ops: compiled code, step-boundary work, GPU idle.
usage: op_timing_report.py DIR"""
import collections
import csv
import glob
import os
import sys

for f in sorted(glob.glob(os.path.join(sys.argv[1], "rank*.csv")))[:4]:
    rows = [r for r in csv.DictReader(open(f)) if r["step"] != "step"]
    # group by prompt
    prompts, cur, last_t = [], [], None
    for r in rows:
        t = float(r["host_t"])
        if last_t is not None and t - last_t > 1.5:
            prompts.append(cur); cur = []
        cur.append(r); last_t = t
    prompts.append(cur)
    print(f"\n===== {os.path.basename(f)}: {len(rows)} op rows, {len(prompts)} prompts")
    groups = collections.defaultdict(list)
    for p in prompts:
        toks = sum(int(n) for n in {r["step"]: r["num_tokens"] for r in p}.values())
        if toks >= 40000: groups["long (~85k+)"].append(p)
        elif toks >= 8000: groups["~13k"].append(p)
    for label, ps in groups.items():
        dur = collections.Counter(); kids = collections.Counter(); calls = collections.Counter(); other = 0.0; bigother = 0.0
        steps = set(); toks = {}
        for p in ps:
            for r in p:
                k = (id(p), r["step"]); steps.add(k); toks[k] = int(r["num_tokens"])
                d = float(r["dur_ms"]); dur[r["name"]] += d; calls[r["name"]] += 1
                if r["parent"]:
                    kids[r["parent"]] += d
                else:
                    g = float(r["gap_ms"])
                    if g > 0:
                        other += g
                        if g > 40: bigother += g
        n = len(steps); selft = {k: dur[k] - kids[k] for k in dur}; tot = sum(selft.values()) + other
        print(f"\n--- {label}: {len(ps)} prompts, {n} steps, {sum(toks.values())} tokens, {tot / n:.0f} ms/step measured")
        print(f"{'op':44}{'calls/step':>11}{'ms/step':>10}{'%':>7}{'ms/call':>9}")
        for k, v in sorted(selft.items(), key=lambda x: -x[1])[:32]:
            print(f"{k[:43]:44}{calls[k] / n:>11.1f}{v / n:>10.1f}{100 * v / tot:>7.1f}{v / max(calls[k], 1):>9.3f}")
        print(f"{'(other: compiled code, boundaries, idle)':44}{'':>11}{other / n:>10.1f}{100 * other / tot:>7.1f}   of which gaps > 40 ms: {bigother / n:.1f} ms/step")
    if os.environ.get("ONE_RANK", "1") == "1":
        break
