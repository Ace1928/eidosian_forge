from numba.extending import typeof_impl
from numba.extending import models, register_model
from numba.extending import unbox, NativeValue, box
from numba.core.imputils import lower_constant, lower_cast
from numba.core.ccallback import CFunc
from numba.core import cgutils
from llvmlite import ir
from numba.core import types
from numba.core.types import (FunctionType, UndefinedFunctionType,
from numba.core.dispatcher import Dispatcher
def lower_get_wrapper_address(context, builder, func, sig, failure_mode='return_exc'):
    """Low-level call to _get_wrapper_address(func, sig).

    When calling this function, GIL must be acquired.
    """
    pyapi = context.get_python_api(builder)
    modname = context.insert_const_string(builder.module, __name__)
    numba_mod = pyapi.import_module_noblock(modname)
    numba_func = pyapi.object_getattr_string(numba_mod, '_get_wrapper_address')
    pyapi.decref(numba_mod)
    sig_obj = pyapi.unserialize(pyapi.serialize_object(sig))
    addr = pyapi.call_function_objargs(numba_func, (func, sig_obj))
    if failure_mode != 'ignore':
        with builder.if_then(cgutils.is_null(builder, addr), likely=False):
            if failure_mode == 'return_exc':
                context.call_conv.return_exc(builder)
            elif failure_mode == 'return_null':
                builder.ret(pyapi.get_null_object())
            else:
                raise NotImplementedError(failure_mode)
    return addr