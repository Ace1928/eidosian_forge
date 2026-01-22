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
def initialize_dim3(builder, prefix):
    x = nvvmutils.call_sreg(builder, '%s.x' % prefix)
    y = nvvmutils.call_sreg(builder, '%s.y' % prefix)
    z = nvvmutils.call_sreg(builder, '%s.z' % prefix)
    return cgutils.pack_struct(builder, (x, y, z))