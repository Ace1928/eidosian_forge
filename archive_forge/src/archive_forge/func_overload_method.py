import os
import uuid
import weakref
import collections
import functools
import numba
from numba.core import types, errors, utils, config
from numba.core.typing.typeof import typeof_impl  # noqa: F401
from numba.core.typing.asnumbatype import as_numba_type  # noqa: F401
from numba.core.typing.templates import infer, infer_getattr  # noqa: F401
from numba.core.imputils import (  # noqa: F401
from numba.core.datamodel import models   # noqa: F401
from numba.core.datamodel import register_default as register_model  # noqa: F401, E501
from numba.core.pythonapi import box, unbox, reflect, NativeValue  # noqa: F401
from numba._helperlib import _import_cython_function  # noqa: F401
from numba.core.serialize import ReduceMixin
def overload_method(typ, attr, **kwargs):
    """
    A decorator marking the decorated function as typing and implementing
    method *attr* for the given Numba type in nopython mode.

    *kwargs* are passed to the underlying `@overload` call.

    Here is an example implementing .take() for array types::

        @overload_method(types.Array, 'take')
        def array_take(arr, indices):
            if isinstance(indices, types.Array):
                def take_impl(arr, indices):
                    n = indices.shape[0]
                    res = np.empty(n, arr.dtype)
                    for i in range(n):
                        res[i] = arr[indices[i]]
                    return res
                return take_impl
    """
    return _overload_method_common(typ, attr, **kwargs)