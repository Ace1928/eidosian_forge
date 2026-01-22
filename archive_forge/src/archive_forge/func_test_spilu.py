import pytest
import numpy as np
import numpy.testing as npt
import scipy.sparse
import scipy.sparse.linalg as spla
from scipy._lib._util import VisibleDeprecationWarning
def test_spilu():
    X = scipy.sparse.csc_array([[1, 0, 0, 0], [2, 1, 0, 0], [3, 2, 1, 0], [4, 3, 2, 1]])
    LU = spla.spilu(X)
    npt.assert_allclose(LU.solve(np.array([1, 2, 3, 4])), np.asarray([1, 0, 0, 0], dtype=np.float64), rtol=1e-14, atol=3e-16)