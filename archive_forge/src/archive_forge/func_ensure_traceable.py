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
def ensure_traceable(func: Callable[..., R]) -> Callable[..., R]:
    """Ensure that a function is traceable."""
    return cast(SupportsLangsmithExtra, func if is_traceable_function(func) else traceable()(func))