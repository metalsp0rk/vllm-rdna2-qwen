#!/usr/bin/env python3
"""Decode vs prefill overlap (fork VLLM_RDNA_OVERLAP_STEP_TOKENS): how fast does a running decode go while a long
prefill runs next to it, and how much slower does that prefill get?

Per round:
  solo     one uncached prompt of --size tokens with nothing else running: time to first token (= prefill)
  overlap  a streaming decode (short prompt, ignore_eos) runs; after --lead s a second uncached --size prompt arrives.
           Reports the decode rate before the prefill and during it (token timestamps inside [B start, B done]),
           and the prefill's wall time against solo.
Every prompt carries a fresh nonce, so the prefix cache cannot serve it. Uses /v1/completions (no chat template).
usage: overlap_bench.py [--size 16000] [--rounds 2] [--warmup] [--label P=256]
"""
import argparse
import json
import threading
import time
import urllib.request
import uuid

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="http://127.0.0.1:8000")
ap.add_argument("--model", default="qwen38-flash-next")
ap.add_argument("--size", type=int, default=16000, help="approximate prompt tokens of the long prefill")
ap.add_argument("--rounds", type=int, default=2)
ap.add_argument("--lead", type=float, default=4.0, help="seconds of steady decode before the prefill arrives")
ap.add_argument("--warmup", action="store_true", help="one small overlap first (JIT / first-shape compiles)")
ap.add_argument("--label", default="")
a = ap.parse_args()


def post(path, body, stream=False, timeout=1800):
    req = urllib.request.Request(a.base + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def filler(n_tokens, u):
    return " ".join(f"{u}-{i} item {i} value {i * 7}" for i in range(int(n_tokens / 1.3) // 6 + 1))


def prefill(n_tokens, gen=32):
    """Streams one uncached prompt: returns (start, first-token time, prompt tokens, continuation text with the
    nonce masked). The continuation of the filler pattern is predictable, so solo and overlapped runs (different
    chunking) must produce the same masked text."""
    u = uuid.uuid4().hex
    body = {"model": a.model, "prompt": filler(n_tokens, u), "max_tokens": gen, "temperature": 0, "stream": True, "ignore_eos": True,
            "stream_options": {"include_usage": True}}
    t0 = time.perf_counter(); first = None; text = ""; pt = 0
    with post("/v1/completions", body) as r:
        for line in r:
            line = line.strip()
            if not line.startswith(b"data:") or line == b"data: [DONE]":
                continue
            j = json.loads(line[5:])
            if j.get("usage"):
                pt = j["usage"]["prompt_tokens"]
            for c in j.get("choices", []):
                if c.get("text"):
                    first = first or time.perf_counter()
                    text += c["text"]
    return t0, first, pt, text.replace(u, "X")


class Decoder(threading.Thread):
    """Streams a long generation and records the arrival time of every text chunk (~1 token each)."""

    def __init__(self, max_tokens=4000):
        super().__init__(daemon=True)
        self.stamps, self.stop, self.max_tokens = [], False, max_tokens

    def run(self):
        body = {"model": a.model, "prompt": f"[{uuid.uuid4().hex}] Write a long story about a lighthouse keeper.",
                "max_tokens": self.max_tokens, "temperature": 0, "ignore_eos": True, "stream": True}
        with post("/v1/completions", body) as r:
            for line in r:
                if self.stop:
                    break
                line = line.strip()
                if not line.startswith(b"data:") or line == b"data: [DONE]":
                    continue
                if json.loads(line[5:])["choices"][0].get("text"):
                    self.stamps.append(time.perf_counter())

    def rate(self, t0, t1):
        n = sum(1 for t in self.stamps if t0 <= t <= t1)
        return n / (t1 - t0) if t1 > t0 else 0.0, n

    def max_gap(self, t0, t1):
        s = [t for t in self.stamps if t0 <= t <= t1]
        return max((b - x for x, b in zip(s, s[1:])), default=t1 - t0)


def overlap(n_tokens):
    d = Decoder()
    d.start()
    while len(d.stamps) < 5:
        time.sleep(0.05)
    time.sleep(a.lead)
    t0, t1, pt, text = prefill(n_tokens)
    time.sleep(1.0)
    d.stop = True
    base_rate, _ = d.rate(t0 - a.lead + 1.0, t0)
    ov_rate, ov_n = d.rate(t0, t1)
    gap = d.max_gap(t0, t1)
    d.join(timeout=30)
    return pt, t1 - t0, base_rate, ov_rate, ov_n, gap, text


tag = f"[{a.label}] " if a.label else ""
if a.warmup:
    overlap(2000)
    prefill(2000)
for r in range(a.rounds):
    s0, s1, spt, stext = prefill(a.size)
    solo = s1 - s0
    pt, wall, base_rate, ov_rate, ov_n, gap, otext = overlap(a.size)
    # the nonce eats most of the 32 tokens and can be spelled with different token splits, so the visible suffix
    # length varies; consistent = one continuation is a prefix of the other
    short, long_ = sorted((stext, otext), key=len)
    same = ("consistent" if long_.startswith(short) else
            f"DIFFERENT: solo {stext[:60]!r} vs overlapped {otext[:60]!r}")
    print(f"{tag}round {r + 1}: prompt {pt} tok | solo TTFT {solo:.2f} s ({spt / solo:.0f} tok/s) | "
          f"overlapped TTFT {wall:.2f} s ({pt / wall:.0f} tok/s, {wall / solo:.2f}x solo) | "
          f"decode before {base_rate:.1f} tok/s, during {ov_rate:.2f} tok/s ({ov_n} tok, max gap {gap:.2f} s) | "
          f"continuation {same}",
          flush=True)
