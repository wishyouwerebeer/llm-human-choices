"""Deterministic parser for Psych-101 transcripts of the four-armed restless bandit
(bahrami2020four/exp.csv).

A transcript consists of an instruction block, a blank line, and one line per trial:

    You press <<L>> and get 84.0 points.

The human's choice is the token enclosed in << >>; the number is the reward.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

CHOICE_MARKER = re.compile(r"<<(.*?)>>")
TRIAL_LINE = re.compile(r"^You press <<([A-Z])>> and get (-?\d+(?:\.\d+)?) points\.$")
OPTIONS_LINE = re.compile(r"options labeled ([A-Z](?:, [A-Z])*,? and [A-Z])\.")


@dataclass(frozen=True)
class Trial:
    choice: str
    reward: float


@dataclass(frozen=True)
class ParsedTranscript:
    instructions: str
    options: tuple[str, ...]
    trials: tuple[Trial, ...]

    @property
    def choices(self) -> list[str]:
        return [t.choice for t in self.trials]

    @property
    def rewards(self) -> list[float]:
        return [t.reward for t in self.trials]


def parse_options(instructions: str) -> tuple[str, ...]:
    m = OPTIONS_LINE.search(instructions)
    if not m:
        raise ValueError("Could not find option labels in instructions")
    return tuple(re.findall(r"[A-Z]", m.group(1)))


def parse_transcript(text: str) -> ParsedTranscript:
    """Split a transcript into (instructions, options, trials).

    Raises ValueError on any line that does not match the expected trial format,
    so malformed data can never be silently dropped.
    """
    instructions, sep, body = text.strip().partition("\n\n")
    if not sep:
        raise ValueError("Transcript has no blank line separating instructions from trials")
    options = parse_options(instructions)
    trials = []
    for i, line in enumerate(body.strip().split("\n")):
        m = TRIAL_LINE.match(line.strip())
        if not m:
            raise ValueError(f"Unparseable trial line {i}: {line!r}")
        choice, reward = m.group(1), float(m.group(2))
        if choice not in options:
            raise ValueError(f"Choice {choice!r} not among options {options}")
        trials.append(Trial(choice, reward))
    return ParsedTranscript(instructions.strip(), options, tuple(trials))


def format_trial(trial: Trial) -> str:
    """Re-render a trial exactly in the Psych-101 format (used to build LLM prompts)."""
    return f"You press <<{trial.choice}>> and get {trial.reward} points."
