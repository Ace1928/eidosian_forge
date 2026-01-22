import pytest
import numpy as np
import numpy.testing as npt
import scipy.sparse
import scipy.sparse.linalg as spla
from scipy._lib._util import VisibleDeprecationWarning
@parametrize_eig_sparrays
def test_eigsh(X):
    X = X + X.T
    e, v = spla.eigsh(X, k=1)
    npt.assert_allclose(X @ v, e[0] * v)