from __future__ import annotations
import collections.abc as c
import datetime
import typing as t
from .util import (
from .util_common import (
from .metadata import (
from .config import (
from . import junit_xml
def calculate_confidence(path: str, line: int, metadata: Metadata) -> int:
    """Return the confidence level for a test result associated with the given file path and line number."""
    ranges = metadata.changes.get(path)
    if not ranges:
        return 0
    if any((r[0] <= line <= r[1] for r in ranges)):
        return 100
    if line == 0:
        return 75
    return 50