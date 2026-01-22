from __future__ import annotations
import asyncio
import contextlib
import contextvars
import datetime
import functools
import inspect
import logging
import traceback
import uuid
import warnings
from contextvars import copy_context
from typing import (
from langsmith import client as ls_client
from langsmith import run_trees, utils
from langsmith._internal import _aiter as aitertools
def is_traceable_function(func: Callable) -> bool:
    """Check if a function is @traceable decorated."""
    return _is_traceable_function(func) or (isinstance(func, functools.partial) and _is_traceable_function(func.func)) or (hasattr(func, '__call__') and _is_traceable_function(func.__call__))