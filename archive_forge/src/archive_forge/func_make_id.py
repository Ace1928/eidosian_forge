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
def make_id() -> ID:
    """ Return a new unique ID for a Bokeh object.

    Normally this function will return simple monotonically increasing integer
    IDs (as strings) for identifying Bokeh objects within a Document. However,
    if it is desirable to have globally unique for every object, this behavior
    can be overridden by setting the environment variable ``BOKEH_SIMPLE_IDS=no``.

    Returns:
        str

    """
    global _simple_id
    if settings.simple_ids():
        with _simple_id_lock:
            _simple_id += 1
            return ID(f'p{_simple_id}')
    else:
        return make_globally_unique_id()