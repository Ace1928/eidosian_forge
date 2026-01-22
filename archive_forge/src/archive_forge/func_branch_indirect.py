import contextlib
import functools
from llvmlite.ir import instructions, types, values
def branch_indirect(self, addr):
    """
        Indirect branch to target *addr*.
        """
    br = instructions.IndirectBranch(self.block, 'indirectbr', addr)
    self._set_terminator(br)
    return br