from __future__ import annotations
import sys
from typing import Any, Type
import inspect
from contextlib import contextmanager
from functools import cmp_to_key, update_wrapper
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.function import AppliedUndef, UndefinedFunction, Function
def print_function(print_cls):
    """ A decorator to replace kwargs with the printer settings in __signature__ """

    def decorator(f):
        if sys.version_info < (3, 9):
            cls = type(f'{f.__qualname__}_PrintFunction', (_PrintFunction,), {'__doc__': f.__doc__})
        else:
            cls = _PrintFunction
        return cls(f, print_cls)
    return decorator