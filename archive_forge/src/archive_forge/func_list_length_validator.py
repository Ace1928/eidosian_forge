import abc
import math
import re
import warnings
from datetime import date
from decimal import Decimal, InvalidOperation
from enum import Enum
from pathlib import Path
from types import new_class
from typing import (
from uuid import UUID
from weakref import WeakSet
from . import errors
from .datetime_parse import parse_date
from .utils import import_string, update_not_none
from .validators import (
@classmethod
def list_length_validator(cls, v: 'Optional[List[T]]') -> 'Optional[List[T]]':
    if v is None:
        return None
    v = list_validator(v)
    v_len = len(v)
    if cls.min_items is not None and v_len < cls.min_items:
        raise errors.ListMinLengthError(limit_value=cls.min_items)
    if cls.max_items is not None and v_len > cls.max_items:
        raise errors.ListMaxLengthError(limit_value=cls.max_items)
    return v