import itertools
import functools
import sys
import operator
from collections import namedtuple
import numpy as np
import unittest
import warnings
from numba import jit, typeof, njit, typed
from numba.core import errors, types, config
from numba.tests.support import (TestCase, tag, ignore_internal_warnings,
from numba.core.extending import overload_method, box
def test_getattr_non_literal_str(self):

    @njit
    def foo(x, nonliteral_str):
        return getattr(x, nonliteral_str)
    with self.assertRaises(errors.TypingError) as raises:
        foo(1, '__hash__')
    msg = "argument 'name' must be a literal string"
    self.assertIn(msg, str(raises.exception))