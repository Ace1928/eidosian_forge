import contextlib
import functools
from llvmlite.ir import instructions, types, values
def insert_value(self, agg, value, idx, name=''):
    """
        Insert *value* into member number *idx* from aggregate.
        """
    if not isinstance(idx, (tuple, list)):
        idx = [idx]
    instr = instructions.InsertValue(self.block, agg, value, idx, name=name)
    self._insert(instr)
    return instr