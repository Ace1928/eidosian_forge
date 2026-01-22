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
@typeof_impl.register(WrapperAddressProtocol)
@typeof_impl.register(CFunc)
def typeof_function_type(val, c):
    if isinstance(val, CFunc):
        sig = val._sig
    elif isinstance(val, WrapperAddressProtocol):
        sig = val.signature()
    else:
        raise NotImplementedError(f'function type from {type(val).__name__}')
    return FunctionType(sig)