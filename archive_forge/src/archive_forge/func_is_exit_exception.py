from __future__ import annotations
import asyncio  # noqa
import typing
from typing import Any
from typing import Callable
from typing import Coroutine
from typing import TypeVar
def is_exit_exception(e):
    return not isinstance(e, Exception)