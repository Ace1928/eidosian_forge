import contextlib
import functools
from llvmlite.ir import instructions, types, values
@_binop('urem')
def urem(self, lhs, rhs, name=''):
    """
        Unsigned integer remainder:
            name = lhs % rhs
        """