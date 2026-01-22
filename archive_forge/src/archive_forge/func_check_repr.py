import unittest
import numpy as np
from numba.tests.support import TestCase
from numba import typeof
from numba.core import types
from numba.typed import List, Dict
def check_repr(self, val):
    ty = typeof(val)
    ty2 = eval(repr(ty), self.tys_ns)
    self.assertEqual(ty, ty2)