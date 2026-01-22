import itertools
import unittest
from numba import jit
from numba.core.controlflow import CFGraph, ControlFlowAnalysis
from numba.core import types
from numba.core.bytecode import FunctionIdentity, ByteCode, _fix_LOAD_GLOBAL_arg
from numba.core.utils import PYVERSION
from numba.tests.support import TestCase
def loopless2(self):
    """
        A loopless CFG corresponding to the following code structure:

            c = (... if ... else ...) + ...
            if c:
                return ...
            else:
                return ...

        Note there are two exit points, and the entry point has been
        changed to a non-zero value.
        """
    g = self.from_adj_list({99: [18, 12], 12: [21], 18: [21], 21: [42, 34], 34: [], 42: []})
    g.set_entry_point(99)
    g.process()
    return g