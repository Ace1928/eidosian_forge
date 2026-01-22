from __future__ import annotations
import re
import typing
from bisect import bisect_right
from dataclasses import dataclass
from zoneinfo import ZoneInfo
import numpy as np
from .breaks import timedelta_helper
from .utils import (
def remove_zeroes(s: str) -> str:
    """
            Remove unnecessary zeros for float string s
            """
    tup = s.split('e')
    if len(tup) == 2:
        mantissa = tup[0].rstrip('0').rstrip('.')
        exponent = int(tup[1])
        s = f'{mantissa}e{exponent}' if exponent else mantissa
    return s