import contextlib
import functools
from llvmlite.ir import instructions, types, values
def position_at_end(self, block):
    """
        Position at the end of the basic *block*.
        """
    self._block = block
    self._anchor = len(block.instructions)