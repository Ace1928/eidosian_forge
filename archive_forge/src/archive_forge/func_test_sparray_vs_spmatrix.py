import os
import numpy as np
import tempfile
from pytest import raises as assert_raises
from numpy.testing import assert_equal, assert_
from scipy.sparse import (sparray, csc_matrix, csr_matrix, bsr_matrix, dia_matrix,
def test_sparray_vs_spmatrix():
    fd, tmpfile = tempfile.mkstemp(suffix='.npz')
    os.close(fd)
    try:
        save_npz(tmpfile, csr_matrix([[1.2, 0, 0.9], [0, 0.3, 0]]))
        loaded_matrix = load_npz(tmpfile)
    finally:
        os.remove(tmpfile)
    fd, tmpfile = tempfile.mkstemp(suffix='.npz')
    os.close(fd)
    try:
        save_npz(tmpfile, csr_array([[1.2, 0, 0.9], [0, 0.3, 0]]))
        loaded_array = load_npz(tmpfile)
    finally:
        os.remove(tmpfile)
    assert not isinstance(loaded_matrix, sparray)
    assert isinstance(loaded_array, sparray)
    assert_(loaded_matrix.dtype == loaded_array.dtype)
    assert_equal(loaded_matrix.toarray(), loaded_array.toarray())