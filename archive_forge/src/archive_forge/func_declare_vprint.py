import itertools
from llvmlite import ir
from numba.core import cgutils, targetconfig
from .cudadrv import nvvm
def declare_vprint(lmod):
    voidptrty = ir.PointerType(ir.IntType(8))
    vprintfty = ir.FunctionType(ir.IntType(32), [voidptrty, voidptrty])
    vprintf = cgutils.get_or_insert_function(lmod, vprintfty, 'vprintf')
    return vprintf