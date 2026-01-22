import warnings
import io
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import pytest
from scipy.interpolate import (
def test_roots_extrapolate_gh_11185():
    x = np.array([0.001, 0.002])
    y = np.array([1.66066935e-06, 1.10410807e-06])
    dy = np.array([-1.60061854, -1.600619])
    p = CubicHermiteSpline(x, y, dy)
    r = p.roots(extrapolate=True)
    assert_equal(p.c.shape[1], 1)
    assert_equal(r.size, 3)