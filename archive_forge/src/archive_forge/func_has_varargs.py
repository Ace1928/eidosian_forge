from functools import reduce, partial
import inspect
import sys
from operator import attrgetter, not_
from importlib import import_module
from types import MethodType
from .utils import no_default
from . import _signatures as _sigs
def has_varargs(func, sigspec=None):
    sigspec, rv = _check_sigspec(sigspec, func, _sigs._has_varargs, func)
    if sigspec is None:
        return rv
    return any((p.kind == p.VAR_POSITIONAL for p in sigspec.parameters.values()))