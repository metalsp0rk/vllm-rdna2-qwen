# SPDX-License-Identifier: Apache-2.0
"""Measurement hook (gfx1030 fork, 2026-11-05): GPU time per custom op on prefill steps. VLLM_RDNA_OP_TIMING=DIR, off by default.

Custom ops are opaque to torch.compile, so their Python runs at step time even in compiled forwards. Each is wrapped with HIP timing
events on the model stream, read back only after completion, so the stream never blocks. rocprofv3 stalls in-server prefill; this needs no
queue interposition. Writes DIR/rank<device>.csv: one row per op with its parent (nested ops), duration, and for top-level ops the gap since
the previous top-level op ended, which is the compiled code and any idle time between ops. Only steps of at least MIN_TOKENS tokens, set by
set_step() from the PLE connector, are recorded; graph capture is never touched.
"""
import functools
import os
import time
from collections import deque
from contextlib import nullcontext

import torch

_DIR = os.getenv("VLLM_RDNA_OP_TIMING", "")
MIN_TOKENS = int(os.getenv("VLLM_RDNA_OP_TIMING_MIN_TOKENS", "1024"))


class _State:
    def __init__(self) -> None:
        self.active = False
        self.step = 0
        self.num_tokens = 0
        self.stack: list[str] = []
        self.prev_end = None
        self.pending: deque = deque()
        self.f = None

    def event(self):
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        return ev

    def drain(self) -> None:
        if self.f is None:
            os.makedirs(_DIR, exist_ok=True)
            self.f = open(os.path.join(_DIR, f"rank{torch.cuda.current_device()}.csv"), "a", buffering=1 << 16)
            self.f.write("step,host_t,num_tokens,name,parent,gap_ms,dur_ms\n")
        while self.pending and self.pending[0][-1].query():
            step, t, n, name, parent, prev, start, end = self.pending.popleft()
            gap = prev.elapsed_time(start) if prev is not None else -1.0
            self.f.write(f"{step},{t:.3f},{n},{name},{parent},{gap:.4f},{start.elapsed_time(end):.4f}\n")


_S = _State()


def set_step(num_tokens: int) -> None:
    """Called once per model step. Does nothing unless the hook is on."""
    if not _DIR:
        return
    _S.step += 1
    _S.num_tokens = num_tokens
    _S.active = num_tokens >= MIN_TOKENS
    if not _S.active:
        _S.prev_end = None
    elif _S.pending:
        _S.drain()


class _Region:
    """Times a stage inside an op, with the same rows as a wrapped op."""

    __slots__ = ("name", "start", "parent")

    def __init__(self, name: str) -> None:
        self.name = name
        self.start = None

    def __enter__(self):
        s = _S
        if s.active and not torch.cuda.is_current_stream_capturing():
            self.parent = s.stack[-1] if s.stack else ""
            self.start = s.event()
            s.stack.append(self.name)
        return self

    def __exit__(self, *exc):
        if self.start is not None:
            s = _S
            s.stack.pop()
            s.pending.append((s.step, time.time(), s.num_tokens, self.name, self.parent, None, self.start, s.event()))
        return False


_NOOP = nullcontext()


def region(name: str):
    return _Region(name) if _DIR else _NOOP


def wrap(name: str, fn):
    """The op itself when the hook is off, so production has no extra call."""
    if not _DIR:
        return fn

    @functools.wraps(fn)
    def timed(*args, **kwargs):
        if not _S.active or torch.cuda.is_current_stream_capturing():
            return fn(*args, **kwargs)
        s = _S
        parent = s.stack[-1] if s.stack else ""
        start = s.event()
        s.stack.append(name)
        try:
            out = fn(*args, **kwargs)
        finally:
            s.stack.pop()
        end = s.event()
        s.pending.append((s.step, time.time(), s.num_tokens, name, parent, None if parent else s.prev_end, start, end))
        if not parent:
            s.prev_end = end
            if len(s.pending) >= 512:
                s.drain()
        return out

    return timed
