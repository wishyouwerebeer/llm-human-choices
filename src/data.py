"""Loading the bahrami2020four subset of Psych-101 and sampling participants.

Only the rows of the selected experiment are cached locally (data/, gitignored);
the full ~860 MB corpus is streamed once and never written to disk.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .parser import ParsedTranscript, parse_transcript

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
EXPERIMENT = "bahrami2020four/exp.csv"
LOCAL_FILE = DATA_DIR / "bahrami2020four.jsonl"
HF_DATASET = "marcelbinz/Psych-101"

N_PARTICIPANTS = 20
SEED = 42
MIN_TRIALS = 140
N_EVAL = 20


def download_subset(path: Path = LOCAL_FILE) -> None:
    from datasets import load_dataset

    path.parent.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(HF_DATASET, split="train", streaming=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in ds:
            if row["experiment"] == EXPERIMENT:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.rename(path)


def load_rows(path: Path = LOCAL_FILE) -> list[dict]:
    if not path.exists():
        print(f"Streaming {HF_DATASET} to extract {EXPERIMENT} (one-time) ...")
        download_subset(path)
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


@dataclass(frozen=True)
class Participant:
    pid: str
    data: ParsedTranscript

    @property
    def n_trials(self) -> int:
        return len(self.data.trials)

    @property
    def split(self) -> int:
        """Index of the first second-half trial (50/50 split)."""
        return self.n_trials // 2

    @property
    def eval_indices(self) -> list[int]:
        """First N_EVAL trials of the second half (0-based trial indices)."""
        return list(range(self.split, min(self.split + N_EVAL, self.n_trials)))


def sample_participants(n: int = N_PARTICIPANTS, seed: int = SEED,
                        min_trials: int = MIN_TRIALS) -> list[Participant]:
    """Parse all rows, keep participants with >= min_trials, draw n with a fixed seed."""
    rows = load_rows()
    parts = [Participant(r["participant"], parse_transcript(r["text"])) for r in rows]
    eligible = sorted((p for p in parts if p.n_trials >= min_trials), key=lambda p: int(p.pid))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(eligible), size=n, replace=False)
    return [eligible[i] for i in idx]
