import contextlib
import functools
from llvmlite.ir import instructions, types, values
def icmp_signed(self, cmpop, lhs, rhs, name=''):
    """
        Signed integer comparison:
            name = lhs <cmpop> rhs

        where cmpop can be '==', '!=', '<', '<=', '>', '>='
        """
    return self._icmp('s', cmpop, lhs, rhs, name)