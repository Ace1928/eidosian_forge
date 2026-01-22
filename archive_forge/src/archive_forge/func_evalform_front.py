from functools import reduce, partial
import inspect
import sys
from operator import attrgetter, not_
from importlib import import_module
from types import MethodType
from .utils import no_default
from . import _signatures as _sigs
def evalform_front(val, form):
    if callable(form):
        return form(val)
    if isinstance(form, tuple):
        func, args = (form[0], form[1:])
        args = (val,) + args
        return func(*args)