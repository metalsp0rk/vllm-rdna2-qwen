#!/usr/bin/env python3
"""In-server sweep of the QSA sparse attention prefill kernel: one boot, many configs, on real traffic. A one-card bench ranked these configs the opposite way from the server,
so rank them here. Boot with VLLM_RDNA_QSA_TUNE_FILE=TUNEFILE and VLLM_RDNA_OP_TIMING=OUTDIR (create TUNEFILE with the first config first). For each config this script writes the
file, warms the kernel up with two small prompts, then times one ~14k and one ~60k prompt. Repeat the baseline among the configs to see drift. Stop the server, then:
qsa_tune.py --report OUTDIR.
usage: qsa_tune.py OUTDIR TUNEFILE "BLOCK_N:WARPS:STAGES:SPLITS:DOT4" ...    e.g. 32:4:1:1:0 16:4:1:1:1 32:4:1:1:0"""
import bisect
import csv
import glob
import json
import os
import subprocess
import sys
import time

if sys.argv[1] == "--report":
    D = sys.argv[2]
    W = json.load(open(D + "/windows.json"))
    data = []
    for f in sorted(glob.glob(D + "/rank*.csv")):
        rows = [(float(r["host_t"]), r["name"], r["parent"], float(r["dur_ms"]), float(r["gap_ms"]), r["step"]) for r in csv.DictReader(open(f)) if r["step"] != "step"]
        data.append(sorted(rows, key=lambda x: x[0]))

    def stats(t0, t1):
        sp, n, tot, steps = 0.0, 0, 0.0, 0
        for rows in data:
            ts = [x[0] for x in rows]
            seen = set()
            for x in rows[bisect.bisect_left(ts, t0):bisect.bisect_right(ts, t1)]:
                if x[1] == "qsa.sparse":
                    sp += x[3]; n += 1
                if not x[2]:
                    tot += x[3] + max(x[4], 0); seen.add(x[5])
            steps += len(seen)
        return sp / max(n, 1), tot / max(steps, 1)

    rate = lambda o: o.split(", ")[-1].replace(" tok/s", "") if "tok/s" in o else "?"
    print(f"{'config':16}{'14k tok/s':>11}{'sparse ms':>11}{'ms/step':>9}{'60k tok/s':>11}{'sparse ms':>11}{'ms/step':>9}")
    for x in W:
        s, l = stats(*x["short"][:2]), stats(*x["long"][:2])
        print(f"{x['cfg']:16}{rate(x['short'][2]):>11}{s[0]:>11.2f}{s[1]:>9.0f}{rate(x['long'][2]):>11}{l[0]:>11.2f}{l[1]:>9.0f}")
    sys.exit(0)

out, tune, cfgs = sys.argv[1], sys.argv[2], sys.argv[3:]
DRV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prefill_load.py")


def run(toks):
    t0 = time.time()
    o = subprocess.run([sys.executable, DRV, "--tokens", str(toks)], capture_output=True, text=True, timeout=600).stdout.strip()
    return t0, time.time(), o


w = []
for c in cfgs:
    with open(tune + ".tmp", "w") as f:
        f.write(c.replace(":", " "))
    os.replace(tune + ".tmp", tune)
    run(3000)
    run(3000)
    a = run(14000)
    b = run(60000)
    w.append(dict(cfg=c, short=list(a), long=list(b)))
    print(f"{c:16} 14k: {a[2]} | 60k: {b[2]}", flush=True)
    json.dump(w, open(out + "/windows.json", "w"))
