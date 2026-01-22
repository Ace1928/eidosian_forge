import warnings
import sys
import os
import itertools
import pytest
import weakref
import numpy as np
from numpy.testing import (
def test_float64_ignore_nan(self):
    offset = np.uint64(4294967295)
    nan1_i64 = np.array(np.nan, dtype=np.float64).view(np.uint64)
    nan2_i64 = nan1_i64 ^ offset
    nan1_f64 = nan1_i64.view(np.float64)
    nan2_f64 = nan2_i64.view(np.float64)
    assert_array_max_ulp(nan1_f64, nan2_f64, 0)