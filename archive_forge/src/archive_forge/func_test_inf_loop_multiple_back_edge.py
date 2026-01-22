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
def test_inf_loop_multiple_back_edge(self):
    cfunc = self.compile(inf_loop_multiple_back_edge)
    rec = RefRecorder()
    iterator = iter(cfunc(rec))
    next(iterator)
    self.assertEqual(rec.alive, [])
    next(iterator)
    self.assertEqual(rec.alive, [])
    next(iterator)
    self.assertEqual(rec.alive, [])
    self.assertEqual(rec.recorded, ['yield', 'p', 'bra', 'yield', 'p', 'bra', 'yield'])