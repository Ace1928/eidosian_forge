from __future__ import annotations
import asyncio
import inspect
import math
import operator
from collections.abc import Iterable, Iterator
from functools import partial
from types import FunctionType, MethodType
from typing import Any, Callable, Optional
from .depends import depends
from .display import _display_accessors, _reactive_display_objs
from .parameterized import (
from .parameters import Boolean, Event
from ._utils import _to_async_gen, iscoroutinefunction, full_groupby
@classmethod
def register_display_handler(cls, obj_type, handler, **kwargs):
    """
        Registers a display handler for a specific type of object,
        making it possible to define custom display options for
        specific objects.

        Arguments
        ---------
        obj_type: type | callable
          The type to register a custom display handler on.
        handler: Viewable | callable
          A Viewable or callable that is given the object to be displayed
          and the custom keyword arguments.
        kwargs: dict[str, Any]
          Additional display options to register for this type.
        """
    cls._display_handlers[obj_type] = (handler, kwargs)