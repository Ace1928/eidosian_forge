import warnings
import sys
import os
import itertools
import pytest
import weakref
import numpy as np
from numpy.testing import (
def test_nan_item(self):
    self._assert_func(np.nan, np.nan)
    assert_raises(AssertionError, lambda: self._assert_func(np.nan, 1))
    assert_raises(AssertionError, lambda: self._assert_func(np.nan, np.inf))
    assert_raises(AssertionError, lambda: self._assert_func(np.inf, np.nan))