import llvmlite.ir
from llvmlite.ir import Constant, IRBuilder
from numba.core import types, config, cgutils
from numba.core.funcdesc import FunctionDescriptor
def get_args_ptr(self, builder, genptr):
    return cgutils.gep_inbounds(builder, genptr, 0, 1)