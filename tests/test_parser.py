"""Parser tests on 3 real Psych-101 transcripts (bahrami2020four, Apache-2.0)."""
import json
import re
from pathlib import Path

import pytest

from src.parser import format_trial, parse_transcript

FIXTURES = json.loads((Path(__file__).parent / "fixtures" / "bahrami2020four_3_transcripts.json")
                      .read_text(encoding="utf-8"))


@pytest.mark.parametrize("row", FIXTURES, ids=lambda r: f"participant_{r['participant']}")
def test_trial_count_equals_choice_markers(row):
    parsed = parse_transcript(row["text"])
    assert len(parsed.trials) == len(re.findall(r"<<.*?>>", row["text"]))


@pytest.mark.parametrize("row", FIXTURES, ids=lambda r: f"participant_{r['participant']}")
def test_choices_match_markers_in_order(row):
    parsed = parse_transcript(row["text"])
    assert parsed.choices == re.findall(r"<<(.*?)>>", row["text"])


@pytest.mark.parametrize("row", FIXTURES, ids=lambda r: f"participant_{r['participant']}")
def test_options_and_rewards(row):
    parsed = parse_transcript(row["text"])
    assert len(parsed.options) == 4 and len(set(parsed.options)) == 4
    assert set(parsed.choices) <= set(parsed.options)
    assert all(isinstance(r, float) for r in parsed.rewards)
    assert "<<" not in parsed.instructions


@pytest.mark.parametrize("row", FIXTURES, ids=lambda r: f"participant_{r['participant']}")
def test_round_trip_formatting(row):
    """Re-rendered trial lines are identical to the raw transcript lines, so the LLM
    sees history in exactly the Psych-101 format."""
    parsed = parse_transcript(row["text"])
    body = row["text"].strip().split("\n\n", 1)[1].strip().split("\n")
    assert [format_trial(t) for t in parsed.trials] == body


def test_malformed_line_raises():
    text = FIXTURES[0]["text"] + "\nSomething unexpected."
    with pytest.raises(ValueError):
        parse_transcript(text)


def test_known_values_first_fixture():
    parsed = parse_transcript(FIXTURES[0]["text"])
    first_line = FIXTURES[0]["text"].split("\n\n", 1)[1].split("\n")[0]
    m = re.match(r"You press <<(\w)>> and get (-?[\d.]+) points\.", first_line)
    assert parsed.trials[0].choice == m.group(1)
    assert parsed.trials[0].reward == float(m.group(2))


def test_every_trial_line_round_trips_exactly():
    """For every trial in all three fixtures, format_trial reproduces the raw line."""
    n = 0
    for row in FIXTURES:
        lines = row["text"].strip().split("\n\n", 1)[1].strip().split("\n")
        parsed = parse_transcript(row["text"])
        assert len(lines) == len(parsed.trials)
        for i, (trial, line) in enumerate(zip(parsed.trials, lines)):
            assert format_trial(trial) == line, (row["participant"], i, line)
            n += 1
    assert n == 150 + 147 + 141
