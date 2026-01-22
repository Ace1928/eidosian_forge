from __future__ import annotations
from typing import Any
import builtins
import inspect
import keyword
import textwrap
import linecache
from sympy.external import import_module # noqa:F401
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.decorator import doctest_depends_on
from sympy.utilities.iterables import (is_sequence, iterable,
from sympy.utilities.misc import filldedent
def sub_args(args, dummies_dict):
    if isinstance(args, str):
        return args
    elif isinstance(args, DeferredVector):
        return str(args)
    elif iterable(args):
        dummies = flatten([sub_args(a, dummies_dict) for a in args])
        return ','.join((str(a) for a in dummies))
    elif isinstance(args, (Function, Symbol, Derivative)):
        dummies = Dummy()
        dummies_dict.update({args: dummies})
        return str(dummies)
    else:
        return str(args)