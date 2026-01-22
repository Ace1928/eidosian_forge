import typing
import types
import inspect
import functools
from . import _uarray
import copyreg
import pickle
import contextlib
from ._uarray import (  # type: ignore
def pickle_function(func):
    mod_name = getattr(func, '__module__', None)
    qname = getattr(func, '__qualname__', None)
    self_ = getattr(func, '__self__', None)
    try:
        test = unpickle_function(mod_name, qname, self_)
    except pickle.UnpicklingError:
        test = None
    if test is not func:
        raise pickle.PicklingError(f"Can't pickle {func}: it's not the same object as {test}")
    return (unpickle_function, (mod_name, qname, self_))