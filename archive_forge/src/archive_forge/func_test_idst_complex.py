from os.path import join, dirname
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_equal
import pytest
from pytest import raises as assert_raises
from scipy.fftpack._realtransforms import (
def test_idst_complex(self):
    y = idst(np.arange(5) * 1j)
    x = 1j * idst(np.arange(5))
    assert_array_almost_equal(x, y)