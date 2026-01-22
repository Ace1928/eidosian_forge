from functools import reduce, partial
import inspect
import sys
from operator import attrgetter, not_
from importlib import import_module
from types import MethodType
from .utils import no_default
from . import _signatures as _sigs
def memof(*args, **kwargs):
    k = key(args, kwargs)
    try:
        return cache[k]
    except TypeError:
        raise TypeError('Arguments to memoized function must be hashable')
    except KeyError:
        cache[k] = result = func(*args, **kwargs)
        return result