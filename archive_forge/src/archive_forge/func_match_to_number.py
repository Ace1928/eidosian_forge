from __future__ import annotations
from datetime import date, datetime, time, timedelta, timezone, tzinfo
from functools import lru_cache
import re
from typing import Any
from ._types import ParseFloat
def match_to_number(match: re.Match, parse_float: ParseFloat) -> Any:
    if match.group('floatpart'):
        return parse_float(match.group())
    return int(match.group(), 0)