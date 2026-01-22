from __future__ import annotations
import logging # isort:skip
import inspect
import time
from copy import copy
from functools import wraps
from typing import (
from tornado import locks
from ..events import ConnectionLost
from ..util.token import generate_jwt_token
from .callbacks import DocumentCallbackGroup
def current_time() -> float:
    """Return the time in milliseconds since the epoch as a floating
       point number.
    """
    return time.monotonic() * 1000