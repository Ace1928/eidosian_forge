import warnings
import io
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import pytest
from scipy.interpolate import (
def test_small_dx(self):
    rng = np.random.RandomState(0)
    x = np.sort(rng.uniform(size=100))
    y = 10000.0 + rng.uniform(size=100)
    S = CubicSpline(x, y)
    self.check_correctness(S, tol=1e-13)