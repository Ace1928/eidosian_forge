import dis
from pprint import pformat
import logging
from collections import namedtuple, defaultdict, deque
from functools import total_ordering
from numba.core.utils import UniqueDict, PYVERSION, ALL_BINOPS_TO_OPERATORS
from numba.core.controlflow import NEW_BLOCKERS, CFGraph
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError
def op_GEN_START(self, state, inst):
    """Pops TOS. If TOS was not None, raises an exception. The kind
        operand corresponds to the type of generator or coroutine and
        determines the error message. The legal kinds are 0 for generator,
        1 for coroutine, and 2 for async generator.

        New in version 3.10.
        """
    pass