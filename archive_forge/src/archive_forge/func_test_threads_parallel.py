import sys
import threading
import numpy as np
from numpy import array, finfo, arange, eye, all, unique, ones, dot
import numpy.random as random
from numpy.testing import (
import pytest
from pytest import raises as assert_raises
import scipy.linalg
from scipy.linalg import norm, inv
from scipy.sparse import (spdiags, SparseEfficiencyWarning, csc_matrix,
from scipy.sparse.linalg import SuperLU
from scipy.sparse.linalg._dsolve import (spsolve, use_solver, splu, spilu,
import scipy.sparse
from scipy._lib._testutils import check_free_memory
from scipy._lib._util import ComplexWarning
@pytest.mark.slow
@sup_sparse_efficiency
def test_threads_parallel(self):
    oks = []

    def worker():
        try:
            self.test_splu_basic()
            self._internal_test_splu_smoketest()
            self._internal_test_spilu_smoketest()
            oks.append(True)
        except Exception:
            pass
    threads = [threading.Thread(target=worker) for k in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert_equal(len(oks), 20)