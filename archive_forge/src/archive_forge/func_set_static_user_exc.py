from collections import namedtuple
from collections.abc import Iterable
import itertools
import hashlib
from llvmlite import ir
from numba.core import types, cgutils, errors
from numba.core.base import PYOBJECT, GENERIC_POINTER
def set_static_user_exc(self, builder, exc, exc_args=None, loc=None, func_name=None):
    if exc is not None and (not issubclass(exc, BaseException)):
        raise TypeError('exc should be None or exception class, got %r' % (exc,))
    if exc_args is not None and (not isinstance(exc_args, tuple)):
        raise TypeError('exc_args should be None or tuple, got %r' % (exc_args,))
    if exc_args is None:
        exc_args = tuple()
    pyapi = self.context.get_python_api(builder)
    exc = self.build_excinfo_struct(exc, exc_args, loc, func_name)
    struct_gv = pyapi.serialize_object(exc)
    excptr = self._get_excinfo_argument(builder.function)
    store = builder.store(struct_gv, excptr)
    md = builder.module.add_metadata([ir.IntType(1)(1)])
    store.set_metadata('numba_exception_output', md)