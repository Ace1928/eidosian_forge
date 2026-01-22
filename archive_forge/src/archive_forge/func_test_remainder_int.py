import contextlib
import sys
import numpy as np
from numba import vectorize, guvectorize
from numba.tests.support import (TestCase, CheckWarningsMixin,
import unittest
def test_remainder_int(self):
    self.check_divmod_int(remainder, [0, 0, 0, 1])