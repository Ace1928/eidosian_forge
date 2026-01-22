from functools import partial
import importlib
import itertools
import numbers
from six import string_types, iteritems
from ..core import Machine
from .nesting import HierarchicalMachine
def rep(func, format_references=None):
    """ Return a string representation for `func`. """
    if isinstance(func, string_types):
        return func
    if isinstance(func, numbers.Number):
        return str(func)
    return format_references(func) if format_references is not None else None