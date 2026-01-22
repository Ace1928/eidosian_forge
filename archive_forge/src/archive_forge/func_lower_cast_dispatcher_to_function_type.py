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
@lower_cast(types.Dispatcher, FunctionType)
def lower_cast_dispatcher_to_function_type(context, builder, fromty, toty, val):
    toty = toty.get_precise()
    pyapi = context.get_python_api(builder)
    sfunc = cgutils.create_struct_proxy(toty)(context, builder)
    gil_state = pyapi.gil_ensure()
    addr = lower_get_wrapper_address(context, builder, val, toty.signature, failure_mode='return_exc')
    sfunc.addr = pyapi.long_as_voidptr(addr)
    pyapi.decref(addr)
    pyapi.gil_release(gil_state)
    llty = context.get_value_type(types.voidptr)
    sfunc.pyaddr = builder.ptrtoint(val, llty)
    return sfunc._getvalue()