import contextlib
import functools
from llvmlite.ir import instructions, types, values
@_uniop_intrinsic_int('llvm.bitreverse')
def bitreverse(self, cond):
    """
        Reverse the bitpattern of an integer value; for example 0b10110110
        becomes 0b01101101.
        """