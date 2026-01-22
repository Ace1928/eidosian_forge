import contextlib
import functools
from llvmlite.ir import instructions, types, values
@_uniop_intrinsic_with_flag('llvm.ctlz')
def ctlz(self, cond, flag):
    """
        Counts leading zero bits in *value*. Boolean *flag* indicates whether
        the result is defined for ``0``.
        """