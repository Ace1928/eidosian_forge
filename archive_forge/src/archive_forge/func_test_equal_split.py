import numpy as np
import functools
import sys
import pytest
from numpy.lib.shape_base import (
from numpy.testing import (
def test_equal_split(self):
    a = np.arange(10)
    res = split(a, 2)
    desired = [np.arange(5), np.arange(5, 10)]
    compare_results(res, desired)