import os
import numpy as np
import tempfile
from pytest import raises as assert_raises
from numpy.testing import assert_equal, assert_
from scipy.sparse import (sparray, csc_matrix, csr_matrix, bsr_matrix, dia_matrix,
def test_implemented_error():
    x = dok_matrix((2, 3))
    x[0, 1] = 1
    assert_raises(NotImplementedError, save_npz, 'x.npz', x)