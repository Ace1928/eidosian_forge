from functools import partial
from itertools import product
import operator
from pytest import raises as assert_raises, warns
from numpy.testing import assert_, assert_equal
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg._interface as interface
from scipy.sparse._sputils import matrix
def test_dtypes_of_operator_sum():
    mat_complex = np.random.rand(2, 2) + 1j * np.random.rand(2, 2)
    mat_real = np.random.rand(2, 2)
    complex_operator = interface.aslinearoperator(mat_complex)
    real_operator = interface.aslinearoperator(mat_real)
    sum_complex = complex_operator + complex_operator
    sum_real = real_operator + real_operator
    assert_equal(sum_real.dtype, np.float64)
    assert_equal(sum_complex.dtype, np.complex128)