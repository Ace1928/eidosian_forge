import array
import numpy as np
from numba import jit
from numba.tests.support import TestCase, compile_function, MemoryLeakMixin
import unittest
def check_setitem(self, obj):
    for i in range(len(obj)):
        orig = list(obj)
        val = obj[i] // 2 + 1
        setitem_usecase(obj, i, val)
        self.assertEqual(obj[i], val)
        for j, val in enumerate(orig):
            if j != i:
                self.assertEqual(obj[j], val)