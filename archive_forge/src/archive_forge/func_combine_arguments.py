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
def combine_arguments(wargs, wkwargs, asynchronous=False):
    combined_args = []
    for arg in args:
        if hasattr(arg, '_dinfo'):
            arg = eval_function_with_deps(arg)
        elif isinstance(arg, Parameter):
            arg = getattr(arg.owner, arg.name)
        combined_args.append(arg)
    combined_args += list(wargs)
    combined_kwargs = {}
    for kw, arg in kwargs.items():
        if hasattr(arg, '_dinfo'):
            arg = eval_function_with_deps(arg)
        elif isinstance(arg, Parameter):
            arg = getattr(arg.owner, arg.name)
        combined_kwargs[kw] = arg
    for kw, arg in wkwargs.items():
        if asynchronous:
            if kw.startswith('__arg'):
                index = kw[5:]
                if index.isdigit():
                    combined_args[int(index)] = arg
            elif kw.startswith('__kwarg'):
                substring = kw[8:]
                if substring in combined_kwargs:
                    combined_kwargs[substring] = arg
            continue
        elif kw.startswith('__arg') or kw.startswith('__kwarg') or kw.startswith('__fn'):
            continue
        combined_kwargs[kw] = arg
    return (combined_args, combined_kwargs)