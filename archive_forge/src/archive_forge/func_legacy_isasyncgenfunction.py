from __future__ import annotations
import inspect
import signal
import sys
from functools import wraps
from typing import TYPE_CHECKING, Final, Protocol, TypeVar
import attrs
from .._util import is_main_thread
def legacy_isasyncgenfunction(obj: object) -> TypeGuard[Callable[..., types.AsyncGeneratorType[object, object]]]:
    return getattr(obj, '_async_gen_function', None) == id(obj)