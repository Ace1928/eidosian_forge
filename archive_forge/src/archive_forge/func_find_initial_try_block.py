import dis
from pprint import pformat
import logging
from collections import namedtuple, defaultdict, deque
from functools import total_ordering
from numba.core.utils import UniqueDict, PYVERSION, ALL_BINOPS_TO_OPERATORS
from numba.core.controlflow import NEW_BLOCKERS, CFGraph
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError
def find_initial_try_block(self):
    """Find the initial *try* block.
        """
    for blk in reversed(self._blockstack_initial):
        if blk['kind'] == BlockKind('TRY'):
            return blk