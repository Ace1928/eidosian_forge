import warnings
import sys
import os
import itertools
import pytest
import weakref
import numpy as np
from numpy.testing import (
def test_masked_nan_inf(self):
    a = np.ma.MaskedArray([3.0, 4.0, 6.5], mask=[False, True, False])
    b = np.array([3.0, np.nan, 6.5])
    self._test_equal(a, b)
    self._test_equal(b, a)
    a = np.ma.MaskedArray([3.0, 4.0, 6.5], mask=[True, False, False])
    b = np.array([np.inf, 4.0, 6.5])
    self._test_equal(a, b)
    self._test_equal(b, a)