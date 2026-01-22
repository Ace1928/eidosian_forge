import numpy as np
from numpy.testing import (assert_array_equal, assert_equal,
from numpy.lib.arraysetops import (
import pytest
def assert_isin_equal(a, b):
    x = isin(a, b, kind=kind)
    y = isin_slow(a, b)
    assert_array_equal(x, y)