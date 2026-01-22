import pytest
import numpy as np
import numpy.testing as npt
import scipy.sparse
import scipy.sparse.linalg as spla
from scipy._lib._util import VisibleDeprecationWarning
def test_spsolve_triangular():
    X = scipy.sparse.csr_array([[1, 0, 0, 0], [2, 1, 0, 0], [3, 2, 1, 0], [4, 3, 2, 1]])
    spla.spsolve_triangular(X, [1, 2, 3, 4])