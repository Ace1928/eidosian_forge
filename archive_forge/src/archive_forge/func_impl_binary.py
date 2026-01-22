import math
import operator
from llvmlite import ir
from numba.core import types, typing, cgutils, targetconfig
from numba.core.imputils import Registry
from numba.types import float32, float64, int64, uint64
from numba.cuda import libdevice
from numba import cuda
def impl_binary(key, ty, libfunc):
    lower_binary_impl = get_lower_binary_impl(key, ty, libfunc)
    lower(key, ty, ty)(lower_binary_impl)