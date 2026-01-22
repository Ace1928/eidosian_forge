from __future__ import annotations
import contextvars
import sys
import typing as t
from functools import update_wrapper
from types import TracebackType
from werkzeug.exceptions import HTTPException
from . import typing as ft
from .globals import _cv_app
from .globals import _cv_request
from .signals import appcontext_popped
from .signals import appcontext_pushed
def has_app_context() -> bool:
    """Works like :func:`has_request_context` but for the application
    context.  You can also just do a boolean check on the
    :data:`current_app` object instead.

    .. versionadded:: 0.9
    """
    return _cv_app.get(None) is not None