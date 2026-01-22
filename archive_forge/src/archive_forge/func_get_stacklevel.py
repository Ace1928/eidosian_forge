from __future__ import annotations
import inspect
import sys
from importlib import import_module
from inspect import currentframe
from types import CodeType, FrameType, FunctionType
from typing import TYPE_CHECKING, Any, Callable, ForwardRef, Union, cast, final
from weakref import WeakValueDictionary
def get_stacklevel() -> int:
    level = 1
    frame = cast(FrameType, currentframe()).f_back
    while frame and frame.f_globals.get('__name__', '').startswith('typeguard.'):
        level += 1
        frame = frame.f_back
    return level