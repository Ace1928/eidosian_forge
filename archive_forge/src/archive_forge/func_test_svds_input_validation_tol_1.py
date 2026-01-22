import os
import re
import copy
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_array_equal
import pytest
from scipy.linalg import svd, null_space
from scipy.sparse import csc_matrix, issparse, spdiags, random
from scipy.sparse.linalg import LinearOperator, aslinearoperator
from scipy.sparse.linalg import svds
from scipy.sparse.linalg._eigen.arpack import ArpackNoConvergence
@pytest.mark.parametrize('tol', (-1, np.inf, np.nan))
def test_svds_input_validation_tol_1(self, tol):
    message = '`tol` must be a non-negative floating point value.'
    with pytest.raises(ValueError, match=message):
        svds(np.eye(10), tol=tol, solver=self.solver)