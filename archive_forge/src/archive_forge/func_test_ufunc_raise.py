import contextlib
import sys
import numpy as np
from numba import vectorize, guvectorize
from numba.tests.support import (TestCase, CheckWarningsMixin,
import unittest
def test_ufunc_raise(self):
    self.check_ufunc_raise(nopython=True)