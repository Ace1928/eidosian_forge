from __future__ import annotations
import sys
from collections.abc import Callable
from typing import TypeVar
from warnings import warn
from ._core._eventloop import get_async_backend
from .abc import CapacityLimiter

    Return the capacity limiter that is used by default to limit the number of
    concurrent threads.

    :return: a capacity limiter object

    