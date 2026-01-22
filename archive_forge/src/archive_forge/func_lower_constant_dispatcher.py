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
@lower_constant(types.Dispatcher)
def lower_constant_dispatcher(context, builder, typ, pyval):
    return context.add_dynamic_addr(builder, id(pyval), info=type(pyval).__name__)