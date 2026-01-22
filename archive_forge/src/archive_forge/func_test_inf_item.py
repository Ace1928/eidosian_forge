import warnings
import sys
import os
import itertools
import pytest
import weakref
import numpy as np
from numpy.testing import (
def test_inf_item(self):
    self._assert_func(np.inf, np.inf)
    self._assert_func(-np.inf, -np.inf)
    assert_raises(AssertionError, lambda: self._assert_func(np.inf, 1))
    assert_raises(AssertionError, lambda: self._assert_func(-np.inf, np.inf))