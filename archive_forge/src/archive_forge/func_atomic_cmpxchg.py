import itertools
from llvmlite import ir
from numba.core import cgutils, targetconfig
from .cudadrv import nvvm
def atomic_cmpxchg(builder, lmod, isize, ptr, cmp, val):
    out = builder.cmpxchg(ptr, cmp, val, 'monotonic', 'monotonic')
    return builder.extract_value(out, 0)