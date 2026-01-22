import numpy as np
from numpy import array
from numpy.testing import (assert_equal, assert_,
import pytest
from pytest import raises as assert_raises
from scipy._lib._testutils import check_free_memory
from scipy._lib._util import check_random_state
from scipy.sparse import (csr_matrix, coo_matrix,
from scipy.sparse._construct import rand as sprand
def test_block_return_type(self):
    block = construct.block_array
    Fl, Gl = ([[1, 2], [3, 4]], [[7], [5]])
    Fm, Gm = (csr_matrix(Fl), csr_matrix(Gl))
    assert isinstance(block([[None, Fl], [Gl, None]], format='csr'), sparray)
    assert isinstance(block([[None, Fm], [Gm, None]], format='csr'), sparray)
    assert isinstance(block([[Fm, Gm]], format='csr'), sparray)