import numpy as np
from numpy import array
from numpy.testing import (assert_equal, assert_,
import pytest
from pytest import raises as assert_raises
from scipy._lib._testutils import check_free_memory
from scipy._lib._util import check_random_state
from scipy.sparse import (csr_matrix, coo_matrix,
from scipy.sparse._construct import rand as sprand
@pytest.mark.parametrize('block_array', (construct.bmat, construct.block_array))
def test_block_creation(self, block_array):
    A = coo_array([[1, 2], [3, 4]])
    B = coo_array([[5], [6]])
    C = coo_array([[7]])
    D = coo_array((0, 0))
    expected = array([[1, 2, 5], [3, 4, 6], [0, 0, 7]])
    assert_equal(block_array([[A, B], [None, C]]).toarray(), expected)
    E = csr_array((1, 2), dtype=np.int32)
    assert_equal(block_array([[A.tocsr(), B.tocsr()], [E, C.tocsr()]]).toarray(), expected)
    assert_equal(block_array([[A.tocsc(), B.tocsc()], [E.tocsc(), C.tocsc()]]).toarray(), expected)
    expected = array([[1, 2, 0], [3, 4, 0], [0, 0, 7]])
    assert_equal(block_array([[A, None], [None, C]]).toarray(), expected)
    assert_equal(block_array([[A.tocsr(), E.T.tocsr()], [E, C.tocsr()]]).toarray(), expected)
    assert_equal(block_array([[A.tocsc(), E.T.tocsc()], [E.tocsc(), C.tocsc()]]).toarray(), expected)
    Z = csr_array((1, 1), dtype=np.int32)
    expected = array([[0, 5], [0, 6], [7, 0]])
    assert_equal(block_array([[None, B], [C, None]]).toarray(), expected)
    assert_equal(block_array([[E.T.tocsr(), B.tocsr()], [C.tocsr(), Z]]).toarray(), expected)
    assert_equal(block_array([[E.T.tocsc(), B.tocsc()], [C.tocsc(), Z.tocsc()]]).toarray(), expected)
    expected = np.empty((0, 0))
    assert_equal(block_array([[None, None]]).toarray(), expected)
    assert_equal(block_array([[None, D], [D, None]]).toarray(), expected)
    expected = array([[7]])
    assert_equal(block_array([[None, D], [C, None]]).toarray(), expected)
    with assert_raises(ValueError) as excinfo:
        block_array([[A], [B]])
    excinfo.match('Got blocks\\[1,0\\]\\.shape\\[1\\] == 1, expected 2')
    with assert_raises(ValueError) as excinfo:
        block_array([[A.tocsr()], [B.tocsr()]])
    excinfo.match('incompatible dimensions for axis 1')
    with assert_raises(ValueError) as excinfo:
        block_array([[A.tocsc()], [B.tocsc()]])
    excinfo.match('Mismatching dimensions along axis 1: ({1, 2}|{2, 1})')
    with assert_raises(ValueError) as excinfo:
        block_array([[A, C]])
    excinfo.match('Got blocks\\[0,1\\]\\.shape\\[0\\] == 1, expected 2')
    with assert_raises(ValueError) as excinfo:
        block_array([[A.tocsr(), C.tocsr()]])
    excinfo.match('Mismatching dimensions along axis 0: ({1, 2}|{2, 1})')
    with assert_raises(ValueError) as excinfo:
        block_array([[A.tocsc(), C.tocsc()]])
    excinfo.match('incompatible dimensions for axis 0')