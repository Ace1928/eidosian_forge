import contextlib
import sys
import numpy as np
from numba import vectorize, guvectorize
from numba.tests.support import (TestCase, CheckWarningsMixin,
import unittest
@skip_m1_fenv_errors
def test_remainder_float(self):
    self.check_divmod_float(remainder, [0.0, float('nan'), float('nan'), 1.0], ['invalid value encountered'])