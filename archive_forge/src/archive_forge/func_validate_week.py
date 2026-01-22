from __future__ import annotations
from datetime import datetime
from . import util
import re
from . import css_types as ct
import unicodedata
import bs4  # type: ignore[import]
from typing import Iterator, Iterable, Any, Callable, Sequence, cast  # noqa: F401
@staticmethod
def validate_week(year: int, week: int) -> bool:
    """Validate week."""
    max_week = datetime.strptime(f'{12}-{31}-{year}', '%m-%d-%Y').isocalendar()[1]
    if max_week == 1:
        max_week = 53
    return 1 <= week <= max_week