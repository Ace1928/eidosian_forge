import contextlib
import functools
from llvmlite.ir import instructions, types, values
def fcmp_unordered(self, cmpop, lhs, rhs, name='', flags=()):
    """
        Floating-point unordered comparison:
            name = lhs <cmpop> rhs

        where cmpop can be '==', '!=', '<', '<=', '>', '>=', 'ord', 'uno'
        """
    if cmpop in _CMP_MAP:
        op = 'u' + _CMP_MAP[cmpop]
    else:
        op = cmpop
    instr = instructions.FCMPInstr(self.block, op, lhs, rhs, name=name, flags=flags)
    self._insert(instr)
    return instr