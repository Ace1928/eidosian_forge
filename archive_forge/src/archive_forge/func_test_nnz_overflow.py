import sys
import os
import gc
import threading
import numpy as np
from numpy.testing import assert_equal, assert_, assert_allclose
from scipy.sparse import (_sparsetools, coo_matrix, csr_matrix, csc_matrix,
from scipy.sparse._sputils import supported_dtypes
from scipy._lib._testutils import check_free_memory
import pytest
from pytest import raises as assert_raises
@pytest.mark.slow
@pytest.mark.xfail_on_32bit("Can't create large array for test")
def test_nnz_overflow():
    nnz = np.iinfo(np.int32).max + 1
    check_free_memory((4 + 4 + 1) * nnz / 1000000.0 + 0.5)
    row = np.zeros(nnz, dtype=np.int32)
    col = np.zeros(nnz, dtype=np.int32)
    data = np.zeros(nnz, dtype=np.int8)
    data[-1] = 4
    s = coo_matrix((data, (row, col)), shape=(1, 1), copy=False)
    d = s.toarray()
    assert_allclose(d, [[4]])