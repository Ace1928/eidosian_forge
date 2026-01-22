from functools import reduce
import operator
import math
from llvmlite import ir
import llvmlite.binding as ll
from numba.core.imputils import Registry, lower_cast
from numba.core.typing.npydecl import parse_dtype
from numba.core.datamodel import models
from numba.core import types, cgutils
from numba.np import ufunc_db
from numba.np.npyimpl import register_ufuncs
from .cudadrv import nvvm
from numba import cuda
from numba.cuda import nvvmutils, stubs, errors
from numba.cuda.types import dim3, CUDADispatcher
def float16_float_ty_constraint(bitwidth):
    typemap = {32: ('f32', 'f'), 64: ('f64', 'd')}
    try:
        return typemap[bitwidth]
    except KeyError:
        msg = f'Conversion between float16 and float{bitwidth} unsupported'
        raise errors.CudaLoweringError(msg)