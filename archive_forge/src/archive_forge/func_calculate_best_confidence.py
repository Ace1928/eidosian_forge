from __future__ import annotations
import collections.abc as c
import datetime
import typing as t
from .util import (
from .util_common import (
from .metadata import (
from .config import (
from . import junit_xml
def calculate_best_confidence(choices: tuple[tuple[str, int], ...], metadata: Metadata) -> int:
    """Return the best confidence value available from the given choices and metadata."""
    best_confidence = 0
    for path, line in choices:
        confidence = calculate_confidence(path, line, metadata)
        best_confidence = max(confidence, best_confidence)
    return best_confidence