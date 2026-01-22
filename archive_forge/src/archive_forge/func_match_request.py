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
def match_request(self) -> None:
    """Can be overridden by a subclass to hook into the matching
        of the request.
        """
    try:
        result = self.url_adapter.match(return_rule=True)
        self.request.url_rule, self.request.view_args = result
    except HTTPException as e:
        self.request.routing_exception = e