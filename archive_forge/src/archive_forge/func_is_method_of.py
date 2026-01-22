from __future__ import annotations
import inspect
import sys
from importlib import import_module
from inspect import currentframe
from types import CodeType, FrameType, FunctionType
from typing import TYPE_CHECKING, Any, Callable, ForwardRef, Union, cast, final
from weakref import WeakValueDictionary
def is_method_of(obj: object, cls: type) -> bool:
    return inspect.isfunction(obj) and obj.__module__ == cls.__module__ and obj.__qualname__.startswith(cls.__qualname__ + '.')