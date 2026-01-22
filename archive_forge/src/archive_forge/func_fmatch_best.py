from __future__ import annotations
from difflib import SequenceMatcher
from typing import Iterable, Iterator
from kombu import version_info_t
def fmatch_best(needle: str, haystack: Iterable[str], min_ratio: float=0.6) -> str | None:
    """Fuzzy match - Find best match (scalar)."""
    try:
        return sorted(fmatch_iter(needle, haystack, min_ratio), reverse=True)[0][1]
    except IndexError:
        return None