"""Offline tests for prompt construction and response parsing (no Claude calls)."""
import json
from pathlib import Path

import numpy as np

from src.claude_predictor import build_prompt, cache_key, parse_probabilities
from src.parser import parse_transcript

ROW = json.loads((Path(__file__).parent / "fixtures" / "bahrami2020four_3_transcripts.json")
                 .read_text(encoding="utf-8"))[0]
DATA = parse_transcript(ROW["text"])
OPTS = DATA.options


def test_parse_clean_json():
    text = json.dumps({o: p for o, p in zip(OPTS, [0.7, 0.1, 0.1, 0.1])})
    assert np.allclose(parse_probabilities(text, OPTS), [0.7, 0.1, 0.1, 0.1])


def _js(values):
    return json.dumps({o: v for o, v in zip(OPTS, values)})


def test_parse_decimals_are_renormalised():
    p = parse_probabilities("```json\n" + _js([0.5, 0.3, 0.3, 0.1]) + "\n```", OPTS)
    assert np.allclose(p, np.array([0.5, 0.3, 0.3, 0.1]) / 1.2)
    assert np.allclose(parse_probabilities(_js([1, 0, 0, 0]), OPTS), [1, 0, 0, 0])


def test_parse_percentages_are_divided_by_100():
    assert np.allclose(parse_probabilities(_js([70, 10, 10, 10]), OPTS), [0.7, 0.1, 0.1, 0.1])
    p = parse_probabilities(_js([60, 20, 15, 3]), OPTS)  # sums to 98: accepted, renormalised
    assert np.allclose(p, np.array([60, 20, 15, 3]) / 98)


def test_parse_rejects_invalid_values():
    assert parse_probabilities(_js([2.0, 1.0, 1.0, -1.0]), OPTS) is None  # >1 and negative
    assert parse_probabilities(_js([0.6, 0.2, 0.3, -0.1]), OPTS) is None  # negative
    assert parse_probabilities(_js([1.5, 0.2, 0.2, 0.1]), OPTS) is None   # >1, not percentages
    assert parse_probabilities(_js([70, 10, 10, 30]), OPTS) is None       # sums to 120
    assert parse_probabilities(_js([110, -5, 0, -5]), OPTS) is None       # sums to 100, negatives
    assert parse_probabilities(_js(["high", 0.2, 0.2, 0.1]), OPTS) is None


def test_parse_rejects_wrong_keys_and_garbage():
    assert parse_probabilities('{"A": 0.5, "B": 0.5}', OPTS) is None
    assert parse_probabilities("I think L.", OPTS) is None
    assert parse_probabilities("", OPTS) is None
    assert parse_probabilities(json.dumps({o: 0 for o in OPTS}), OPTS) is None


def test_prompt_contains_exactly_history_before_t():
    t = 80
    prompt = build_prompt(DATA, t)
    assert prompt.count("You press <<") == t
    assert DATA.instructions in prompt
    assert f"trial {t + 1}" in prompt
    for o in OPTS:
        assert f'"{o}": <probability>' in prompt


def test_cache_key_depends_on_model_and_prompt():
    p = build_prompt(DATA, 80)
    assert cache_key("haiku", p) == cache_key("haiku", p)
    assert cache_key("haiku", p) != cache_key("sonnet", p)
    assert cache_key("haiku", p) != cache_key("haiku", build_prompt(DATA, 81))


# ---------------------------------------------------------------- caching behaviour
import subprocess  # noqa: E402

import pytest  # noqa: E402

import src.claude_predictor as cp  # noqa: E402


def _fake_cli(responses):
    """Return a stand-in for _run_cli that yields the given (returncode, response) pairs."""
    it = iter(responses)

    def run(prompt, model):
        rc, resp = next(it)
        return {"returncode": rc, "stderr": "", "wall_seconds": 0.01, "response": resp}
    return run


@pytest.fixture
def predictor(tmp_path, monkeypatch):
    monkeypatch.setattr(cp.time, "sleep", lambda s: None)
    return cp.ClaudePredictor(cache_dir=tmp_path)


def test_cli_error_is_not_cached(predictor, tmp_path, monkeypatch):
    err = (1, {"is_error": True, "result": "API Error: connection reset"})
    monkeypatch.setattr(cp, "_run_cli", _fake_cli([err] * 4))
    probs, rec = predictor.predict(DATA, 80)
    assert np.allclose(probs, 0.25) and rec["parse_failure"] and rec.get("cli_error")
    assert list(tmp_path.glob("*.json")) == []


def test_cli_error_stops_retry_loop_and_is_not_a_malformed_retry(predictor, tmp_path,
                                                                  monkeypatch):
    calls = []
    err = (1, {"is_error": True, "result": '{"note": "error page"}'})
    fake = _fake_cli([err] * 4)
    monkeypatch.setattr(cp, "_run_cli", lambda p, m: calls.append(1) or fake(p, m))
    probs, rec = predictor.predict(DATA, 80)
    # _call_with_backoff retries a non-limit error once (2 CLI calls); predict() must not
    # start a second, "malformed output" round on top of that.
    assert len(calls) == 2
    assert len(rec["attempts"]) == 1
    assert predictor.stats["retries_malformed"] == 0
    assert rec["parse_failure"] and rec.get("cli_error") and np.allclose(probs, 0.25)
    assert list(tmp_path.glob("*.json")) == []


def test_malformed_then_cli_error_is_not_cached(predictor, tmp_path, monkeypatch):
    bad = (0, {"is_error": False, "result": "no json here"})
    err = (1, {"is_error": True, "result": "API Error"})
    monkeypatch.setattr(cp, "_run_cli", _fake_cli([bad, err, err]))
    _, rec = predictor.predict(DATA, 80)
    assert len(rec["attempts"]) == 2 and predictor.stats["retries_malformed"] == 1
    assert rec.get("cli_error") and list(tmp_path.glob("*.json")) == []


def test_success_is_cached(predictor, tmp_path, monkeypatch):
    ok = (0, {"is_error": False, "result": _js([0.7, 0.1, 0.1, 0.1])})
    monkeypatch.setattr(cp, "_run_cli", _fake_cli([ok]))
    probs, _ = predictor.predict(DATA, 80)
    assert np.allclose(probs, [0.7, 0.1, 0.1, 0.1])
    assert len(list(tmp_path.glob("*.json"))) == 1
    monkeypatch.setattr(cp, "_run_cli", _fake_cli([]))  # any new call would raise
    probs2, _ = predictor.predict(DATA, 80)
    assert np.allclose(probs, probs2)


def test_malformed_output_is_cached_as_parse_failure(predictor, tmp_path, monkeypatch):
    bad = (0, {"is_error": False, "result": "I think the participant will press L."})
    monkeypatch.setattr(cp, "_run_cli", _fake_cli([bad, bad]))
    probs, rec = predictor.predict(DATA, 80)
    assert np.allclose(probs, 0.25) and rec["parse_failure"] and len(rec["attempts"]) == 2
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1 and json.loads(files[0].read_text())["parse_failure"]


def test_timeout_is_reported_as_error(monkeypatch):
    def boom(*a, **k):
        raise subprocess.TimeoutExpired(cmd="claude", timeout=1)
    monkeypatch.setattr(cp.subprocess, "run", boom)
    out = cp._run_cli("prompt", "haiku")
    assert out["returncode"] != 0 and out["response"]["is_error"]
