from __future__ import annotations
from datetime import datetime
from . import util
import re
from . import css_types as ct
import unicodedata
import bs4  # type: ignore[import]
from typing import Iterator, Iterable, Any, Callable, Sequence, cast  # noqa: F401
@staticmethod
def validate_day(year: int, month: int, day: int) -> bool:
    """Validate day."""
    max_days = LONG_MONTH
    if month == FEB:
        max_days = FEB_LEAP_MONTH if year % 4 == 0 and year % 100 != 0 or year % 400 == 0 else FEB_MONTH
    elif month in MONTHS_30:
        max_days = SHORT_MONTH
    return 1 <= day <= max_days