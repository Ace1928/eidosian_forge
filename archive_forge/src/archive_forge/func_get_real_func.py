from __future__ import annotations
import dataclasses
import enum
import functools
import inspect
from inspect import Parameter
from inspect import signature
import os
from pathlib import Path
import sys
from typing import Any
from typing import Callable
from typing import Final
from typing import NoReturn
import py
def get_real_func(obj):
    """Get the real function object of the (possibly) wrapped object by
    functools.wraps or functools.partial."""
    start_obj = obj
    for i in range(100):
        new_obj = getattr(obj, '__pytest_wrapped__', None)
        if isinstance(new_obj, _PytestWrapper):
            obj = new_obj.obj
            break
        new_obj = getattr(obj, '__wrapped__', None)
        if new_obj is None:
            break
        obj = new_obj
    else:
        from _pytest._io.saferepr import saferepr
        raise ValueError(f'could not find real function of {saferepr(start_obj)}\nstopped at {saferepr(obj)}')
    if isinstance(obj, functools.partial):
        obj = obj.func
    return obj