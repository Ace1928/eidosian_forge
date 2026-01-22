from __future__ import annotations
import os
import time
import uuid
import random
import typing
import logging
import traceback
import contextlib
import contextvars
import asyncio
import functools
import inspect
from concurrent import futures
from lazyops.utils.helpers import import_function, timer, build_batches
from lazyops.utils.ahelpers import amap_v2 as concurrent_map
def get_func_full_name(func: typing.Union[str, typing.Callable]) -> str:
    """
    Returns the function name
    """
    return f'{func.__module__}.{func.__qualname__}' if callable(func) else func