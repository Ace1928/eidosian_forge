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
def is_timedelta_type(obj: Any) -> TypeGuard[dt.timedelta | np.timedelta64]:
    """ Whether an object is any timedelta type recognized by Bokeh.

    Args:
        obj (object) : the object to test

    Returns:
        bool : True if ``obj`` is a timedelta type

    """
    return isinstance(obj, (dt.timedelta, np.timedelta64))