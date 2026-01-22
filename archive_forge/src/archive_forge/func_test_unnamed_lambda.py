import unittest
import inspect
from numba import njit
from numba.tests.support import TestCase
from numba.misc.firstlinefinder import get_func_body_first_lineno
def test_unnamed_lambda(self):
    foo = lambda: 1
    first_def_line = get_func_body_first_lineno(njit(foo))
    self.assertIsNone(first_def_line)