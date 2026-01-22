import collections
import functools
import sys
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError
from numba.core.utils import PYVERSION
def incoming_blocks(self, block):
    """
        Yield (incoming block, number of stack pops) pairs for *block*.
        """
    for i, pops in block.incoming_jumps.items():
        if i in self.liveblocks:
            yield (self.blocks[i], pops)