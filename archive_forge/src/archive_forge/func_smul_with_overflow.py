import contextlib
import functools
from llvmlite.ir import instructions, types, values
@_binop_with_overflow('smul')
def smul_with_overflow(self, lhs, rhs, name=''):
    """
        Signed integer multiplication with overflow:
            name = {result, overflow bit} = lhs * rhs
        """