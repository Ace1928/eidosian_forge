from __future__ import annotations
import collections
import enum
from functools import update_wrapper
import inspect
import itertools
import operator
import re
import sys
import textwrap
import threading
import types
from types import CodeType
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import FrozenSet
from typing import Generic
from typing import Iterator
from typing import List
from typing import Mapping
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import warnings
from . import _collections
from . import compat
from ._has_cy import HAS_CYEXTENSION
from .typing import Literal
from .. import exc
def get_callable_argspec(fn: Callable[..., Any], no_self: bool=False, _is_init: bool=False) -> compat.FullArgSpec:
    """Return the argument signature for any callable.

    All pure-Python callables are accepted, including
    functions, methods, classes, objects with __call__;
    builtins and other edge cases like functools.partial() objects
    raise a TypeError.

    """
    if inspect.isbuiltin(fn):
        raise TypeError("Can't inspect builtin: %s" % fn)
    elif inspect.isfunction(fn):
        if _is_init and no_self:
            spec = compat.inspect_getfullargspec(fn)
            return compat.FullArgSpec(spec.args[1:], spec.varargs, spec.varkw, spec.defaults, spec.kwonlyargs, spec.kwonlydefaults, spec.annotations)
        else:
            return compat.inspect_getfullargspec(fn)
    elif inspect.ismethod(fn):
        if no_self and (_is_init or fn.__self__):
            spec = compat.inspect_getfullargspec(fn.__func__)
            return compat.FullArgSpec(spec.args[1:], spec.varargs, spec.varkw, spec.defaults, spec.kwonlyargs, spec.kwonlydefaults, spec.annotations)
        else:
            return compat.inspect_getfullargspec(fn.__func__)
    elif inspect.isclass(fn):
        return get_callable_argspec(fn.__init__, no_self=no_self, _is_init=True)
    elif hasattr(fn, '__func__'):
        return compat.inspect_getfullargspec(fn.__func__)
    elif hasattr(fn, '__call__'):
        if inspect.ismethod(fn.__call__):
            return get_callable_argspec(fn.__call__, no_self=no_self)
        else:
            raise TypeError("Can't inspect callable: %s" % fn)
    else:
        raise TypeError("Can't inspect callable: %s" % fn)