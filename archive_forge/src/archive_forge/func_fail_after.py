from __future__ import annotations
import time
import uuid
import typing
import random
import inspect
import functools
import datetime
import itertools
import asyncio
import contextlib
import async_lru
import signal
from pathlib import Path
from frozendict import frozendict
from typing import Dict, Callable, List, Any, Union, Coroutine, TypeVar, Optional, TYPE_CHECKING
from lazyops.utils.logs import default_logger
from lazyops.utils.serialization import (
from lazyops.utils.lazy import (
@contextlib.contextmanager
def fail_after(delay: Union[int, float]):
    """
    Creates a timeout for a function
    """

    def signal_handler(signum, frame):
        raise TimeoutError(f'Timed out after {delay}s')
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(int(delay))
    try:
        yield
    finally:
        signal.alarm(0)