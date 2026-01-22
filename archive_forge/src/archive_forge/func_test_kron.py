import numpy as np
from numpy import array
from numpy.testing import (assert_equal, assert_,
import pytest
from pytest import raises as assert_raises
from scipy._lib._testutils import check_free_memory
from scipy._lib._util import check_random_state
from scipy.sparse import (csr_matrix, coo_matrix,
from scipy.sparse._construct import rand as sprand
def test_kron(self):
    cases = []
    cases.append(array([[0]]))
    cases.append(array([[-1]]))
    cases.append(array([[4]]))
    cases.append(array([[10]]))
    cases.append(array([[0], [0]]))
    cases.append(array([[0, 0]]))
    cases.append(array([[1, 2], [3, 4]]))
    cases.append(array([[0, 2], [5, 0]]))
    cases.append(array([[0, 2, -6], [8, 0, 14]]))
    cases.append(array([[5, 4], [0, 0], [6, 0]]))
    cases.append(array([[5, 4, 4], [1, 0, 0], [6, 0, 8]]))
    cases.append(array([[0, 1, 0, 2, 0, 5, 8]]))
    cases.append(array([[0.5, 0.125, 0, 3.25], [0, 2.5, 0, 0]]))
    for a in cases:
        ca = csr_array(a)
        for b in cases:
            cb = csr_array(b)
            expected = np.kron(a, b)
            for fmt in sparse_formats[1:4]:
                result = construct.kron(ca, cb, format=fmt)
                assert_equal(result.format, fmt)
                assert_array_equal(result.toarray(), expected)
                assert isinstance(result, sparray)
    a = cases[-1]
    b = cases[-3]
    ca = csr_array(a)
    cb = csr_array(b)
    expected = np.kron(a, b)
    for fmt in sparse_formats:
        result = construct.kron(ca, cb, format=fmt)
        assert_equal(result.format, fmt)
        assert_array_equal(result.toarray(), expected)
        assert isinstance(result, sparray)
    result = construct.kron(csr_matrix(a), csr_matrix(b), format=fmt)
    assert_equal(result.format, fmt)
    assert_array_equal(result.toarray(), expected)
    assert isinstance(result, spmatrix)