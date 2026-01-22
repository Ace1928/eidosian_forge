import numpy as np
from numpy import array
from numpy.testing import (assert_equal, assert_,
import pytest
from pytest import raises as assert_raises
from scipy._lib._testutils import check_free_memory
from scipy._lib._util import check_random_state
from scipy.sparse import (csr_matrix, coo_matrix,
from scipy.sparse._construct import rand as sprand
def test_block_diag_1(self):
    """ block_diag with one matrix """
    assert_equal(construct.block_diag([[1, 0]]).toarray(), array([[1, 0]]))
    assert_equal(construct.block_diag([[[1, 0]]]).toarray(), array([[1, 0]]))
    assert_equal(construct.block_diag([[[1], [0]]]).toarray(), array([[1], [0]]))
    assert_equal(construct.block_diag([1]).toarray(), array([[1]]))