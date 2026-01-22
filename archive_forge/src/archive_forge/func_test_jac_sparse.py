from numpy.testing import assert_
import pytest
from scipy.optimize import _nonlin as nonlin, root
from scipy.sparse import csr_array
from numpy import diag, dot
from numpy.linalg import inv
import numpy as np
from .test_minpack import pressure_network
def test_jac_sparse(self):
    A = csr_array([[1, 2], [2, 1]])
    b = np.array([1, -1])
    self._check_autojac(A, b)
    self._check_autojac((1 + 2j) * A, (2 + 2j) * b)