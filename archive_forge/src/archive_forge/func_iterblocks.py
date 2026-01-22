import collections
import functools
import sys
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError
from numba.core.utils import PYVERSION
def iterblocks(self):
    """
        Return all blocks in sequence of occurrence
        """
    for i in self.blockseq:
        yield self.blocks[i]