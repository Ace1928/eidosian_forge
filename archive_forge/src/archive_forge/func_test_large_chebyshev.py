import warnings
import io
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import pytest
from scipy.interpolate import (
def test_large_chebyshev(self):
    n = 1100
    j = np.arange(n + 1).astype(np.float64)
    x = np.cos(j * np.pi / n)
    w = (-1) ** j
    w[0] *= 0.5
    w[-1] *= 0.5
    P = BarycentricInterpolator(x)
    factor = P.wi[0]
    assert_almost_equal(P.wi / (2 * factor), w)