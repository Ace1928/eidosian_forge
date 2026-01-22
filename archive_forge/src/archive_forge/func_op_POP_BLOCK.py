import dis
from pprint import pformat
import logging
from collections import namedtuple, defaultdict, deque
from functools import total_ordering
from numba.core.utils import UniqueDict, PYVERSION, ALL_BINOPS_TO_OPERATORS
from numba.core.controlflow import NEW_BLOCKERS, CFGraph
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError
def op_POP_BLOCK(self, state, inst):
    blk = state.pop_block()
    if blk['kind'] == BlockKind('TRY'):
        state.append(inst, kind='try')
    elif blk['kind'] == BlockKind('WITH'):
        state.append(inst, kind='with')
    state.fork(pc=inst.next)