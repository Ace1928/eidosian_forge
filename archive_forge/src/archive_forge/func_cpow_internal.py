import math
import operator
from llvmlite import ir
from numba.core import types, typing, cgutils, targetconfig
from numba.core.imputils import Registry
from numba.types import float32, float64, int64, uint64
from numba.cuda import libdevice
from numba import cuda
def cpow_internal(a, b):
    if b.real == fty(0.0) and b.imag == fty(0.0):
        return cty(1.0) + cty(0j)
    elif a.real == fty(0.0) and b.real == fty(0.0):
        return cty(0.0) + cty(0j)
    vabs = math.hypot(a.real, a.imag)
    len = math.pow(vabs, b.real)
    at = math.atan2(a.imag, a.real)
    phase = at * b.real
    if b.imag != fty(0.0):
        len /= math.exp(at * b.imag)
        phase += b.imag * math.log(vabs)
    return len * (cty(math.cos(phase)) + cty(math.sin(phase) * cty(1j)))