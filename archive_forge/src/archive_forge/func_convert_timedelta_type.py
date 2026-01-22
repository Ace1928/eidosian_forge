from __future__ import annotations
import logging # isort:skip
import datetime as dt
import uuid
from functools import lru_cache
from threading import Lock
from typing import TYPE_CHECKING, Any
import numpy as np
from ..core.types import ID
from ..settings import settings
from .strings import format_docstring
def convert_timedelta_type(obj: dt.timedelta | np.timedelta64) -> float:
    """ Convert any recognized timedelta value to floating point absolute
    milliseconds.

    Args:
        obj (object) : the object to convert

    Returns:
        float : milliseconds

    """
    if isinstance(obj, dt.timedelta):
        return obj.total_seconds() * 1000.0
    elif isinstance(obj, np.timedelta64):
        return float(obj / NP_MS_DELTA)
    raise ValueError(f'Unknown timedelta object: {obj!r}')