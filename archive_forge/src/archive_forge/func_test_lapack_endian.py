import warnings
import numpy as np
from numpy import linalg, arange, float64, array, dot, transpose
from numpy.testing import (
def test_lapack_endian(self):
    a = array([[5.7998084, -2.1825367], [-2.1825367, 9.85910595]], dtype='>f8')
    b = array(a, dtype='<f8')
    ap = linalg.cholesky(a)
    bp = linalg.cholesky(b)
    assert_array_equal(ap, bp)