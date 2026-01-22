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
def monkeypatch_proxied_specials(into_cls, from_cls, skip=None, only=None, name='self.proxy', from_instance=None):
    """Automates delegation of __specials__ for a proxying type."""
    if only:
        dunders = only
    else:
        if skip is None:
            skip = ('__slots__', '__del__', '__getattribute__', '__metaclass__', '__getstate__', '__setstate__')
        dunders = [m for m in dir(from_cls) if m.startswith('__') and m.endswith('__') and (not hasattr(into_cls, m)) and (m not in skip)]
    for method in dunders:
        try:
            maybe_fn = getattr(from_cls, method)
            if not hasattr(maybe_fn, '__call__'):
                continue
            maybe_fn = getattr(maybe_fn, '__func__', maybe_fn)
            fn = cast(types.FunctionType, maybe_fn)
        except AttributeError:
            continue
        try:
            spec = compat.inspect_getfullargspec(fn)
            fn_args = compat.inspect_formatargspec(spec[0])
            d_args = compat.inspect_formatargspec(spec[0][1:])
        except TypeError:
            fn_args = '(self, *args, **kw)'
            d_args = '(*args, **kw)'
        py = 'def %(method)s%(fn_args)s: return %(name)s.%(method)s%(d_args)s' % locals()
        env: Dict[str, types.FunctionType] = from_instance is not None and {name: from_instance} or {}
        exec(py, env)
        try:
            env[method].__defaults__ = fn.__defaults__
        except AttributeError:
            pass
        setattr(into_cls, method, env[method])