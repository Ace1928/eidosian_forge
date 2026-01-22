import warnings
import io
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import pytest
from scipy.interpolate import (
def test_all_zeros(self):
    x = np.arange(10)
    y = np.zeros_like(x)
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        pch = pchip(x, y)
    xx = np.linspace(0, 9, 101)
    assert_equal(pch(xx), 0.0)