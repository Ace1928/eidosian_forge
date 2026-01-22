import contextlib
import functools
from llvmlite.ir import instructions, types, values
def extract_value(self, agg, idx, name=''):
    """
        Extract member number *idx* from aggregate.
        """
    if not isinstance(idx, (tuple, list)):
        idx = [idx]
    instr = instructions.ExtractValue(self.block, agg, idx, name=name)
    self._insert(instr)
    return instr