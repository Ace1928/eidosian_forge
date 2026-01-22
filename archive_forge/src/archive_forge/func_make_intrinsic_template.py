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
def make_intrinsic_template(handle, defn, name, *, prefer_literal=False, kwargs=None):
    """
    Make a template class for a intrinsic handle *handle* defined by the
    function *defn*.  The *name* is used for naming the new template class.
    """
    kwargs = MappingProxyType({} if kwargs is None else kwargs)
    base = _IntrinsicTemplate
    name = '_IntrinsicTemplate_%s' % name
    dct = dict(key=handle, _definition_func=staticmethod(defn), _impl_cache={}, _overload_cache={}, prefer_literal=prefer_literal, metadata=kwargs)
    return type(base)(name, (base,), dct)