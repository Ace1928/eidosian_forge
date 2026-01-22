import sys
import numpy as np
from numpy.core._rational_tests import rational
import pytest
from numpy.testing import (
def test_copyto_fromscalar():
    a = np.arange(6, dtype='f4').reshape(2, 3)
    np.copyto(a, 1.5)
    assert_equal(a, 1.5)
    np.copyto(a.T, 2.5)
    assert_equal(a, 2.5)
    mask = np.array([[0, 1, 0], [0, 0, 1]], dtype='?')
    np.copyto(a, 3.5, where=mask)
    assert_equal(a, [[2.5, 3.5, 2.5], [2.5, 2.5, 3.5]])
    mask = np.array([[0, 1], [1, 1], [1, 0]], dtype='?')
    np.copyto(a.T, 4.5, where=mask)
    assert_equal(a, [[2.5, 4.5, 4.5], [4.5, 4.5, 3.5]])