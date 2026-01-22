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
def exercise_generator(self, genfunc):
    cfunc = self.compile(genfunc)
    rec = RefRecorder()
    with self.assertRefCount(rec):
        gen = cfunc(rec)
        next(gen)
        self.assertTrue(rec.alive)
        list(gen)
        self.assertFalse(rec.alive)
    rec = RefRecorder()
    with self.assertRefCount(rec):
        gen = cfunc(rec)
        del gen
        gc.collect()
        self.assertFalse(rec.alive)
    rec = RefRecorder()
    with self.assertRefCount(rec):
        gen = cfunc(rec)
        next(gen)
        self.assertTrue(rec.alive)
        del gen
        gc.collect()
        self.assertFalse(rec.alive)