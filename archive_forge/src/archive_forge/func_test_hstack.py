import numpy as np
from numpy import array
from numpy.testing import (assert_equal, assert_,
import pytest
from pytest import raises as assert_raises
from scipy._lib._testutils import check_free_memory
from scipy._lib._util import check_random_state
from scipy.sparse import (csr_matrix, coo_matrix,
from scipy.sparse._construct import rand as sprand
@pytest.mark.parametrize('coo_cls', [coo_matrix, coo_array])
def test_hstack(self, coo_cls):
    A = coo_cls([[1, 2], [3, 4]])
    B = coo_cls([[5], [6]])
    expected = array([[1, 2, 5], [3, 4, 6]])
    assert_equal(construct.hstack([A, B]).toarray(), expected)
    assert_equal(construct.hstack([A, B], dtype=np.float32).dtype, np.float32)
    assert_equal(construct.hstack([A.tocsc(), B.tocsc()]).toarray(), expected)
    assert_equal(construct.hstack([A.tocsc(), B.tocsc()], dtype=np.float32).dtype, np.float32)
    assert_equal(construct.hstack([A.tocsr(), B.tocsr()]).toarray(), expected)
    assert_equal(construct.hstack([A.tocsr(), B.tocsr()], dtype=np.float32).dtype, np.float32)