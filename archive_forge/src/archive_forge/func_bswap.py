import contextlib
import functools
from llvmlite.ir import instructions, types, values
@_uniop_intrinsic_int('llvm.bswap')
def bswap(self, cond):
    """
        Used to byte swap integer values with an even number of bytes (positive
        multiple of 16 bits)
        """