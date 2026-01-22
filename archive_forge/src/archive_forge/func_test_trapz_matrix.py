import pytest
import textwrap
import warnings
import numpy as np
from numpy.testing import (assert_, assert_equal, assert_raises,
def test_trapz_matrix():
    x = np.linspace(0, 5)
    y = x * x
    r = np.trapz(y, x)
    mx = np.matrix(x)
    my = np.matrix(y)
    mr = np.trapz(my, mx)
    assert_almost_equal(mr, r)