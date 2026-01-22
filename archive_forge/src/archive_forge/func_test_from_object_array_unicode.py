import pytest
import numpy as np
from numpy.core.multiarray import _vec_string
from numpy.testing import (
def test_from_object_array_unicode(self):
    A = np.array([['abc', 'Sigma Σ'], ['long   ', '0123456789']], dtype='O')
    assert_raises(ValueError, np.char.array, (A,))
    B = np.char.array(A, **kw_unicode_true)
    assert_equal(B.dtype.itemsize, 10 * np.array('a', 'U').dtype.itemsize)
    assert_array_equal(B, [['abc', 'Sigma Σ'], ['long', '0123456789']])