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
def build_batches(iterable: typing.Iterable[T], size: int, fixed_batch_size: bool=True) -> typing.Iterable[typing.List[T]]:
    """
    Builds batches of a given size from an iterable.
    """
    if fixed_batch_size:
        return split_into_batches_of_n(iterable, size)
    return split_into_n_batches(iterable, size)