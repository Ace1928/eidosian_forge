import contextlib
import functools
from llvmlite.ir import instructions, types, values
@_uniop_intrinsic_int('llvm.ctpop')
def ctpop(self, cond):
    """
        Counts the number of bits set in a value.
        """