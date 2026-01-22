import contextlib
import functools
from llvmlite.ir import instructions, types, values
def extract_element(self, vector, idx, name=''):
    """
        Returns the value at position idx.
        """
    instr = instructions.ExtractElement(self.block, vector, idx, name=name)
    self._insert(instr)
    return instr