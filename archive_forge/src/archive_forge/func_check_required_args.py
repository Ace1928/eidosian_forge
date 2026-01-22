import functools
import inspect
import itertools
import operator
from importlib import import_module
from .functoolz import (is_partial_args, is_arity, has_varargs,
import builtins
def check_required_args(sig):
    num_pos_only, func, keyword_exclude, sigspec = sig
    return num_required_args(func, sigspec)