import unittest
import inspect
from numba import njit
from numba.tests.support import TestCase
from numba.misc.firstlinefinder import get_func_body_first_lineno
def test_undecorated_odd_comment_indent(self):

    def foo():
        return 1
    first_def_line = get_func_body_first_lineno(njit(foo))
    self.assert_line_location(first_def_line, 3)