"""Claude as a zero-shot, in-context predictor of the next human choice.

Each prediction is one non-interactive `claude -p` call (Claude Code CLI, using the
user's subscription login), with no tools, no MCP servers, no settings/CLAUDE.md,
a minimal custom system prompt, and run from an empty temporary directory.
Responses are cached on disk keyed by SHA-256(model + system prompt + prompt).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

from .parser import ParsedTranscript, format_trial

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / ".cache" / "claude"
MODEL_ALIAS = "haiku"
CLI_TIMEOUT_SECONDS = 300

SYSTEM_PROMPT = (
    "You are an expert in human decision making. You predict what a specific human "
    "participant in a psychology experiment will do next. You answer only with a JSON object."
)


class UsageLimitReached(RuntimeError):
    pass


def build_prompt(data: ParsedTranscript, t: int) -> str:
    """Instructions + the participant's actual history of trials 0..t-1 + the question."""
    opts = data.options
    history = "\n".join(format_trial(tr) for tr in data.trials[:t])
    fmt = "{" + ", ".join(f'"{o}": <probability>' for o in opts) + "}"
    return (
        "Below are the instructions a human participant received in a psychology experiment, "
        "followed by the participant's choices and outcomes so far "
        "(the participant's key presses are enclosed in << >>).\n\n"
        "=== EXPERIMENT TRANSCRIPT ===\n"
        f"{data.instructions}\n\n{history}\n"
        "=== END OF TRANSCRIPT ===\n\n"
        f"This is trial {t + 1}. Predict how likely this particular human participant is to press "
        f"each option ({', '.join(opts)}) on this trial. "
        f"Respond with ONLY a JSON object of the form {fmt}, where the probabilities are "
        "numbers between 0 and 1 that sum to 1. Do not include any other text."
    )


def cache_key(model: str, prompt: str) -> str:
    return hashlib.sha256(f"{model}\n{SYSTEM_PROMPT}\n{prompt}".encode()).hexdigest()


def _to_distribution(p: np.ndarray) -> np.ndarray | None:
    """Values in [0, 1]: renormalise. Values summing to 95-105 with none negative:
    treat as percentages (divide by 100, renormalise). Anything else: reject."""
    if np.all((p >= 0) & (p <= 1)):
        pass
    elif np.all(p >= 0) and 95 <= p.sum() <= 105:
        p = p / 100.0
    else:
        return None
    if p.sum() <= 0:
        return None
    return p / p.sum()


def parse_probabilities(text: str, options: tuple[str, ...]) -> np.ndarray | None:
    """Extract a JSON object from model text, validate keys and values, renormalise.
    Returns None if no valid distribution can be extracted."""
    if not text:
        return None
    candidates = re.findall(r"\{[^{}]*\}", text)
    for cand in reversed(candidates):
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        norm = {str(k).strip().strip("<>").upper(): v for k, v in obj.items()}
        if set(norm) != set(options):
            continue
        try:
            p = np.array([float(norm[o]) for o in options])
        except (TypeError, ValueError):
            continue
        if not np.all(np.isfinite(p)):
            continue
        dist = _to_distribution(p)
        if dist is not None:
            return dist
    return None


_LIMIT_PAT = re.compile(r"usage limit|rate limit|limit reached|hit your limit|429|overloaded", re.I)


def _run_cli(prompt: str, model: str) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")}
    cmd = [
        "claude", "-p",
        "--model", model,
        "--output-format", "json",
        "--max-turns", "1",
        "--tools", "",
        "--strict-mcp-config",
        "--setting-sources", "",
        "--no-session-persistence",
        "--system-prompt", SYSTEM_PROMPT,
    ]
    with tempfile.TemporaryDirectory() as empty_dir:
        t0 = time.time()
        try:
            proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True,
                                  cwd=empty_dir, env=env, timeout=CLI_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            return {"returncode": -1, "stderr": "timeout", "wall_seconds": time.time() - t0,
                    "response": {"is_error": True,
                                 "result": f"CLI timed out after {CLI_TIMEOUT_SECONDS} s"}}
        wall = time.time() - t0
    out = {"returncode": proc.returncode, "stderr": proc.stderr[-2000:], "wall_seconds": wall}
    try:
        out["response"] = json.loads(proc.stdout)
    except json.JSONDecodeError:
        out["response"] = {"is_error": True, "result": proc.stdout[-2000:]}
    return out


class ClaudePredictor:
    def __init__(self, model: str = MODEL_ALIAS, cache_dir: Path = CACHE_DIR,
                 max_limit_retries: int = 4):
        self.model = model
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_limit_retries = max_limit_retries
        self._lock = threading.Lock()
        self.stats = {"calls": 0, "cache_hits": 0, "retries_malformed": 0,
                      "parse_failures": 0, "call_seconds": []}

    def _call_with_backoff(self, prompt: str) -> dict:
        delay = 30
        for attempt in range(self.max_limit_retries + 1):
            out = _run_cli(prompt, self.model)
            resp = out["response"]
            with self._lock:
                self.stats["calls"] += 1
                self.stats["call_seconds"].append(out["wall_seconds"])
            if not resp.get("is_error") and out["returncode"] == 0:
                return out
            msg = f"{resp.get('result', '')} {resp.get('api_error_status', '')} {out['stderr']}"
            if _LIMIT_PAT.search(msg):
                if re.search(r"usage limit|limit reached|hit your limit", msg, re.I) or \
                        attempt == self.max_limit_retries:
                    raise UsageLimitReached(msg.strip()[:500])
                time.sleep(delay)
                delay *= 2
                continue
            if attempt >= 1:
                return out  # non-limit error twice: give up, counted as parse failure
            time.sleep(5)
        return out

    def predict(self, data: ParsedTranscript, t: int) -> tuple[np.ndarray, dict]:
        """Return (probabilities over data.options, record). Uses cache when present."""
        prompt = build_prompt(data, t)
        key = cache_key(self.model, prompt)
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            rec = json.loads(path.read_text())
            with self._lock:
                self.stats["cache_hits"] += 1
        else:
            attempts = []
            probs = None
            cli_error = False
            for _ in range(2):  # original + one retry if malformed
                out = self._call_with_backoff(prompt)
                resp = out["response"]
                cli_error = bool(resp.get("is_error")) or out["returncode"] != 0
                text = resp.get("result", "") if isinstance(resp, dict) else ""
                attempts.append({
                    "text": text,
                    "wall_seconds": out["wall_seconds"],
                    "duration_ms": resp.get("duration_ms"),
                    "usage": resp.get("usage"),
                    "model_usage": list((resp.get("modelUsage") or {}).keys()),
                    "is_error": resp.get("is_error"),
                })
                if cli_error:
                    # Transport/CLI failure (already retried inside _call_with_backoff):
                    # stop here; it is not a malformed output and is not cached.
                    probs = None
                    break
                probs = parse_probabilities(text, data.options)
                if probs is not None:
                    break
                with self._lock:
                    self.stats["retries_malformed"] += 1
            rec = {
                "model_alias": self.model,
                "reported_models": sorted({m for a in attempts for m in a["model_usage"]}),
                "options": list(data.options),
                "trial_index": t,
                "attempts": attempts,
                "probs": None if probs is None else probs.tolist(),
                "parse_failure": probs is None,
            }
            # Cache successful responses and genuine malformed outputs only. If the final
            # attempt was a CLI/transport error (timeout, non-zero exit, is_error), the
            # trial falls back to uniform for this run but is retried on the next run.
            if probs is not None or not cli_error:
                tmp = path.with_suffix(".tmp")
                tmp.write_text(json.dumps(rec, indent=1))
                tmp.rename(path)
            else:
                rec["cli_error"] = True
        if rec["parse_failure"]:
            with self._lock:
                self.stats["parse_failures"] += 1
            k = len(data.options)
            return np.full(k, 1.0 / k), rec
        return np.array(rec["probs"]), rec
