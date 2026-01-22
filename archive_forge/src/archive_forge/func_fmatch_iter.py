from __future__ import annotations
from difflib import SequenceMatcher
from typing import Iterable, Iterator
from kombu import version_info_t
def fmatch_iter(needle: str, haystack: Iterable[str], min_ratio: float=0.6) -> Iterator[tuple[float, str]]:
    """Fuzzy match: iteratively.

    Yields
    ------
        Tuple: of ratio and key.
    """
    for key in haystack:
        ratio = SequenceMatcher(None, needle, key).ratio()
        if ratio >= min_ratio:
            yield (ratio, key)