import contextlib
import functools
from llvmlite.ir import instructions, types, values
@_binop('fadd')
def fadd(self, lhs, rhs, name=''):
    """
        Floating-point addition:
            name = lhs + rhs
        """