from llvmlite import ir
from numba import cuda, types
from numba.core import cgutils
from numba.core.errors import RequireLiteralValue
from numba.core.typing import signature
from numba.core.extending import overload_attribute
from numba.cuda import nvvmutils
from numba.cuda.extending import intrinsic
@overload_attribute(types.Module(cuda), 'warpsize', target='cuda')
def cuda_warpsize(mod):
    """
    The size of a warp. All architectures implemented to date have a warp size
    of 32.
    """

    def get(mod):
        return _warpsize()
    return get