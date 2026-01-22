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
def convert_date_to_datetime(obj: dt.date) -> float:
    """ Convert a date object to a datetime

    Args:
        obj (date) : the object to convert

    Returns:
        datetime

    """
    return (dt.datetime(*obj.timetuple()[:6], tzinfo=dt.timezone.utc) - DT_EPOCH).total_seconds() * 1000