import numpy as np
from numpy import array
from numpy.testing import (assert_equal, assert_,
import pytest
from pytest import raises as assert_raises
from scipy._lib._testutils import check_free_memory
from scipy._lib._util import check_random_state
from scipy.sparse import (csr_matrix, coo_matrix,
from scipy.sparse._construct import rand as sprand
def test_vstack_matrix_or_array(self):
    A = [[1, 2], [3, 4]]
    B = [[5, 6]]
    assert isinstance(construct.vstack([coo_array(A), coo_array(B)]), sparray)
    assert isinstance(construct.vstack([coo_array(A), coo_matrix(B)]), sparray)
    assert isinstance(construct.vstack([coo_matrix(A), coo_array(B)]), sparray)
    assert isinstance(construct.vstack([coo_matrix(A), coo_matrix(B)]), spmatrix)