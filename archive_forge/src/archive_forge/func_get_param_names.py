import ast
import builtins
import dis
import enum
import inspect
import re
import typing
import warnings
from textwrap import dedent
from typing import Type
import torch
from torch._C import (
from torch._sources import get_source_lines_and_file
from .._jit_internal import (  # type: ignore[attr-defined]
from ._state import _get_script_class
from torch._ops import OpOverloadPacket
def get_param_names(fn, n_args):
    if isinstance(fn, OpOverloadPacket):
        fn = fn.op
    if not is_function_or_method(fn) and callable(fn) and is_function_or_method(fn.__call__):
        fn = fn.__call__
    if is_function_or_method(fn):
        if is_ignored_fn(fn):
            fn = inspect.unwrap(fn)
        return inspect.getfullargspec(fn).args
    else:
        return [str(i) for i in range(n_args)]