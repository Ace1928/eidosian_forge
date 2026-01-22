from collections import namedtuple
import numpy as np
from llvmlite.ir import Constant, IRBuilder
from llvmlite import ir
from numba.core import types, cgutils
from numba.core.compiler_lock import global_compiler_lock
from numba.core.caching import make_library_cache, NullCache
def build_obj_loop_body(context, func, builder, arrays, out, offsets, store_offset, signature, pyapi, envptr, env):
    env_body = context.get_env_body(builder, envptr)
    env_manager = pyapi.get_env_manager(env, env_body, envptr)

    def load():
        elems = [ary.load_direct(builder.load(off)) for off, ary in zip(offsets, arrays)]
        elems = [pyapi.from_native_value(t, v, env_manager) for v, t in zip(elems, signature.args)]
        return elems

    def store(retval):
        is_ok = cgutils.is_not_null(builder, retval)
        with builder.if_then(is_ok, likely=True):
            native = pyapi.to_native_value(signature.return_type, retval)
            assert native.cleanup is None
            out.store_direct(native.value, builder.load(store_offset))
            pyapi.decref(retval)
    return _build_ufunc_loop_body_objmode(load, store, context, func, builder, arrays, out, offsets, store_offset, signature, envptr, pyapi)