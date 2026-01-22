import re
from functools import cached_property
import llvmlite.binding as ll
from llvmlite import ir
from numba.core import (cgutils, config, debuginfo, itanium_mangler, types,
from numba.core.dispatcher import Dispatcher
from numba.core.base import BaseContext
from numba.core.callconv import BaseCallConv, MinimalCallConv
from numba.core.typing import cmathdecl
from numba.core import datamodel
from .cudadrv import nvvm
from numba.cuda import codegen, nvvmutils, ufuncs
from numba.cuda.models import cuda_data_manager
@cached_property
def nonconst_module_attrs(self):
    """
        Some CUDA intrinsics are at the module level, but cannot be treated as
        constants, because they are loaded from a special register in the PTX.
        These include threadIdx, blockDim, etc.
        """
    from numba import cuda
    nonconsts = ('threadIdx', 'blockDim', 'blockIdx', 'gridDim', 'laneid', 'warpsize')
    nonconsts_with_mod = tuple([(types.Module(cuda), nc) for nc in nonconsts])
    return nonconsts_with_mod