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
def register_jitable(*args, **kwargs):
    """
    Register a regular python function that can be executed by the python
    interpreter and can be compiled into a nopython function when referenced
    by other jit'ed functions.  Can be used as::

        @register_jitable
        def foo(x, y):
            return x + y

    Or, with compiler options::

        @register_jitable(_nrt=False) # disable runtime allocation
        def foo(x, y):
            return x + y

    """

    def wrap(fn):
        inline = kwargs.pop('inline', 'never')

        @overload(fn, jit_options=kwargs, inline=inline, strict=False)
        def ov_wrap(*args, **kwargs):
            return fn
        return fn
    if kwargs:
        return wrap
    else:
        return wrap(*args)