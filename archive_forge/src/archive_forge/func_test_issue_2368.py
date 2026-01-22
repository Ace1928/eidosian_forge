from io import StringIO
import numpy as np
from numba.core import types
from numba.core.compiler import compile_extra, Flags
from numba.tests.support import TestCase, tag, MemoryLeakMixin
import unittest
def test_issue_2368(self):
    from numba import jit

    def lift_issue2368(a, b):
        s = 0
        for e in a:
            s += e
        h = b.__hash__()
        return (s, h)
    a = np.ones(10)
    b = object()
    jitted = jit(forceobj=True)(lift_issue2368)
    expected = lift_issue2368(a, b)
    got = jitted(a, b)
    self.assertEqual(expected[0], got[0])
    self.assertEqual(expected[1], got[1])
    jitloop = jitted.overloads[jitted.signatures[0]].lifted[0]
    [loopcres] = jitloop.overloads.values()
    self.assertTrue(loopcres.fndesc.native)