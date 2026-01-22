import itertools
from llvmlite import ir
from numba.core import cgutils, targetconfig
from .cudadrv import nvvm
def declare_atomic_add_float64(lmod):
    flags = targetconfig.ConfigStack().top()
    if flags.compute_capability >= (6, 0):
        fname = 'llvm.nvvm.atomic.load.add.f64.p0f64'
    else:
        fname = '___numba_atomic_double_add'
    fnty = ir.FunctionType(ir.DoubleType(), (ir.PointerType(ir.DoubleType()), ir.DoubleType()))
    return cgutils.get_or_insert_function(lmod, fnty, fname)