from os.path import join, dirname
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_equal
import pytest
from pytest import raises as assert_raises
from scipy.fftpack._realtransforms import (
def test_dst_complex64(self):
    y = dst(np.arange(5, dtype=np.complex64) * 1j)
    x = 1j * dst(np.arange(5))
    assert_array_almost_equal(x, y)