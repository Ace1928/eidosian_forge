import pytest
import numpy as np
from numpy.core.multiarray import _vec_string
from numpy.testing import (
def test_from_unicode(self):
    A = np.char.array('Σ')
    assert_equal(len(A), 1)
    assert_equal(len(A[0]), 1)
    assert_equal(A.itemsize, 4)
    assert_(issubclass(A.dtype.type, np.str_))