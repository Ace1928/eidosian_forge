from collections import namedtuple
from collections.abc import Iterable
import itertools
import hashlib
from llvmlite import ir
from numba.core import types, cgutils, errors
from numba.core.base import PYOBJECT, GENERIC_POINTER
def set_dynamic_user_exc(self, builder, exc, exc_args, nb_types, loc=None, func_name=None):
    """
        Compute the required bits to emit an exception with dynamic (runtime)
        values
        """
    if not issubclass(exc, BaseException):
        raise TypeError('exc should be an exception class, got %r' % (exc,))
    if exc_args is not None and (not isinstance(exc_args, tuple)):
        raise TypeError('exc_args should be None or tuple, got %r' % (exc_args,))
    pyapi = self.context.get_python_api(builder)
    exc = self.build_excinfo_struct(exc, exc_args, loc, func_name)
    excinfo_pp = self._get_excinfo_argument(builder.function)
    struct_gv = builder.load(pyapi.serialize_object(exc))
    struct_type = ir.LiteralStructType([arg.type for arg in exc_args if isinstance(arg, ir.Value)])
    st_ptr = self.emit_wrap_args_insts(builder, pyapi, struct_type, exc_args)
    unwrap_fn = self.emit_unwrap_dynamic_exception_fn(builder.module, struct_type, nb_types)
    exc_size = pyapi.py_ssize_t(self.context.get_abi_sizeof(excinfo_t))
    excinfo_p = builder.bitcast(self.context.nrt.allocate(builder, exc_size), excinfo_ptr_t)
    zero = int32_t(0)
    exc_fields = (builder.extract_value(struct_gv, PICKLE_BUF_IDX), builder.extract_value(struct_gv, PICKLE_BUFSZ_IDX), builder.bitcast(st_ptr, GENERIC_POINTER), builder.bitcast(unwrap_fn, GENERIC_POINTER), int32_t(len(struct_type)))
    for idx, arg in enumerate(exc_fields):
        builder.store(arg, builder.gep(excinfo_p, [zero, int32_t(idx)]))
    builder.store(excinfo_p, excinfo_pp)