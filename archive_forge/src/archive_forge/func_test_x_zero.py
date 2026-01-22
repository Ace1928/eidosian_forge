import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import scipy.special as sc
from scipy.special._testutils import FuncData
def test_x_zero(self):
    a = np.arange(1, 10)
    assert_array_equal(sc.gammaincc(a, 0), 1)