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
def make_globally_unique_css_safe_id() -> ID:
    """ Return a globally unique CSS-safe UUID.

    Some situations, e.g. id'ing dynamically created Divs in HTML documents,
    always require globally unique IDs. ID generated with this function can
    be used in APIs like ``document.querySelector("#id")``.

    Returns:
        str

    """
    max_iter = 100
    for _i in range(0, max_iter):
        id = make_globally_unique_id()
        if id[0].isalpha():
            return id
    return ID(f'bk-{make_globally_unique_id()}')