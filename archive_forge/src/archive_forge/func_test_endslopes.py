import warnings
import io
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import pytest
from scipy.interpolate import (
def test_endslopes(self):
    x = np.array([0.0, 0.1, 0.25, 0.35])
    y1 = np.array([279.35, 500.0, 1000.0, 2500.0])
    y2 = np.array([279.35, 2500.0, 1500.0, 1000.0])
    for pp in (pchip(x, y1), pchip(x, y2)):
        for t in (x[0], x[-1]):
            assert_(pp(t, 1) != 0)