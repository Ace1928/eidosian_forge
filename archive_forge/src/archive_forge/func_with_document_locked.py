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
@_needs_document_lock
def with_document_locked(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """ Asynchronously locks the document and runs the function with it locked."""
    return func(*args, **kwargs)