#!/usr/bin/env python3
"""Uncached prefill load: prompts with a fresh nonce, max_tokens=1, /v1/completions.
Prints prompt tokens, seconds and tok/s per prompt. --loop repeats until killed or the server goes away.
usage: prefill_driver.py --tokens 14000 [--count 2] [--loop]
"""
import argparse
import json
import time
import urllib.request
import uuid

ap = argparse.ArgumentParser()
ap.add_argument("--tokens", type=int, required=True)
ap.add_argument("--count", type=int, default=1)
ap.add_argument("--loop", action="store_true")
ap.add_argument("--model", default="qwen38-flash-next")
ap.add_argument("--base", default="http://127.0.0.1:8000")
a = ap.parse_args()
TOK_PER_ENTRY = 46       # about 94k tokens per 2051 "<32-hex nonce>-i item i value 7i" entries

n = 0
while a.loop or n < a.count:
    n += 1
    u = uuid.uuid4().hex
    text = " ".join(f"{u}-{i} item {i} value {i * 7}" for i in range(a.tokens // TOK_PER_ENTRY))
    req = urllib.request.Request(a.base + "/v1/completions", headers={"Content-Type": "application/json"},
                                 data=json.dumps({"model": a.model, "prompt": text, "max_tokens": 1,
                                                  "temperature": 0}).encode())
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            pt = json.load(r)["usage"]["prompt_tokens"]
    except Exception as e:
        print(f"request ended: {type(e).__name__} after {time.perf_counter() - t0:.1f} s", flush=True)
        break
    dt = time.perf_counter() - t0
    print(f"prompt {pt} tok, {dt:.2f} s, {pt / dt:.0f} tok/s", flush=True)
