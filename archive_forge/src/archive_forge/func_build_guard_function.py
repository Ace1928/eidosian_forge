from __future__ import annotations
import ast
import builtins
import collections
import dataclasses
import enum
import functools
import importlib
import inspect
import itertools
import logging
import math
import os
import re
import sys
import textwrap
import types
import weakref
from inspect import currentframe, getframeinfo
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from weakref import ReferenceType
import torch
import torch.utils._device
from torch._dynamo.source import (
from torch._guards import (
from torch.fx.experimental.symbolic_shapes import (
from torch.utils._traceback import format_frame, report_compile_source_on_error
from torch.utils.weak import TensorWeakRef
from . import config, convert_frame, exc, mutation_guard
from .eval_frame import set_guard_error_hook
from .source import DefaultsSource, LocalSource, TypeSource
from .types import GuardedCode, GuardFail, GuardFn  # noqa: F401
from .utils import (
def build_guard_function(code_parts, closure_args) -> Tuple[str, str]:
    from torch._inductor.utils import IndentedBuffer
    if HAS_UNPARSE_FUNCTIONS:
        csepass = PyExprCSEPass()
        csepass.count(code_parts)

        def replace(expr: str) -> Tuple[List[str], str]:
            return csepass.replace(expr)
    else:

        def replace(expr: str) -> Tuple[List[str], str]:
            return ([], expr)
    guard_body = IndentedBuffer()
    for expr in code_parts:
        preface, expr = replace(expr)
        guard_body.writelines(preface)
        guard_body.writeline(f'if not ({expr}):')
        with guard_body.indent():
            guard_body.writeline('return False')
    guard = IndentedBuffer()
    guard.writeline('def guard(L):')
    with guard.indent():
        guard.splice(guard_body)
        guard.writeline('return True')
    make_guard_fn = IndentedBuffer()
    make_guard_fn.writeline(f'def ___make_guard_fn({closure_args}):')
    with make_guard_fn.indent():
        make_guard_fn.splice(guard)
        make_guard_fn.writeline('return guard')
    return (guard_body.getvalue(), make_guard_fn.getvalue())