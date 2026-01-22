import collections
import weakref
import gc
import operator
from itertools import takewhile
import unittest
from numba import njit, jit
from numba.core.compiler import CompilerBase, DefaultPassBuilder
from numba.core.untyped_passes import PreserveIR
from numba.core.typed_passes import IRLegalization
from numba.core import types, ir
from numba.tests.support import TestCase, override_config, SerialMixin
def inf_loop_multiple_back_edge(rec):
    """
    test to reveal bug of invalid liveness when infinite loop has multiple
    backedge.
    """
    while True:
        rec.mark('yield')
        yield
        p = rec('p')
        if p:
            rec.mark('bra')
            pass