import itertools
from llvmlite import ir
from numba.core import cgutils, targetconfig
from .cudadrv import nvvm
def declare_atomic_cas_int(lmod, isize):
    fname = '___numba_atomic_i' + str(isize) + '_cas_hack'
    fnty = ir.FunctionType(ir.IntType(isize), (ir.PointerType(ir.IntType(isize)), ir.IntType(isize), ir.IntType(isize)))
    return cgutils.get_or_insert_function(lmod, fnty, fname)