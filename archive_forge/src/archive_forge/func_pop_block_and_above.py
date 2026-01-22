import dis
from pprint import pformat
import logging
from collections import namedtuple, defaultdict, deque
from functools import total_ordering
from numba.core.utils import UniqueDict, PYVERSION, ALL_BINOPS_TO_OPERATORS
from numba.core.controlflow import NEW_BLOCKERS, CFGraph
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError
def pop_block_and_above(self, blk):
    """Find *blk* in the blockstack and remove it and all blocks above it
        from the stack.
        """
    idx = self._blockstack.index(blk)
    assert 0 <= idx < len(self._blockstack)
    self._blockstack = self._blockstack[:idx]