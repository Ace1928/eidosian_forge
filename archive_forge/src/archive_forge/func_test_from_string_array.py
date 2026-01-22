import pytest
import numpy as np
from numpy.core.multiarray import _vec_string
from numpy.testing import (
def test_from_string_array(self):
    A = np.array([[b'abc', b'foo'], [b'long   ', b'0123456789']])
    assert_equal(A.dtype.type, np.bytes_)
    B = np.char.array(A)
    assert_array_equal(B, A)
    assert_equal(B.dtype, A.dtype)
    assert_equal(B.shape, A.shape)
    B[0, 0] = 'changed'
    assert_(B[0, 0] != A[0, 0])
    C = np.char.asarray(A)
    assert_array_equal(C, A)
    assert_equal(C.dtype, A.dtype)
    C[0, 0] = 'changed again'
    assert_(C[0, 0] != B[0, 0])
    assert_(C[0, 0] == A[0, 0])