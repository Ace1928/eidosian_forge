import functools
import inspect
import itertools
import operator
from importlib import import_module
from .functoolz import (is_partial_args, is_arity, has_varargs,
import builtins
def signature_or_spec(func):
    try:
        return inspect.signature(func)
    except (ValueError, TypeError):
        return None