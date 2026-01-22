from functools import reduce, partial
import inspect
import sys
from operator import attrgetter, not_
from importlib import import_module
from types import MethodType
from .utils import no_default
from . import _signatures as _sigs
def is_valid_args(func, args, kwargs, sigspec=None):
    sigspec, rv = _check_sigspec(sigspec, func, _sigs._is_valid_args, func, args, kwargs)
    if sigspec is None:
        return rv
    try:
        sigspec.bind(*args, **kwargs)
    except TypeError:
        return False
    return True