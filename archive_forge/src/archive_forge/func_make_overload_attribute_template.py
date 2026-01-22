from abc import ABC, abstractmethod
import functools
import sys
import inspect
import os.path
from collections import namedtuple
from collections.abc import Sequence
from types import MethodType, FunctionType, MappingProxyType
import numba
from numba.core import types, utils, targetconfig
from numba.core.errors import (
from numba.core.cpu_options import InlineOptions
def make_overload_attribute_template(typ, attr, overload_func, inline, prefer_literal=False, base=_OverloadAttributeTemplate, **kwargs):
    """
    Make a template class for attribute *attr* of *typ* overloaded by
    *overload_func*.
    """
    assert isinstance(typ, types.Type) or issubclass(typ, types.Type)
    name = 'OverloadAttributeTemplate_%s_%s' % (typ, attr)
    dct = dict(key=typ, _attr=attr, _impl_cache={}, _inline=staticmethod(InlineOptions(inline)), _inline_overloads={}, _overload_func=staticmethod(overload_func), prefer_literal=prefer_literal, metadata=kwargs)
    obj = type(base)(name, (base,), dct)
    return obj