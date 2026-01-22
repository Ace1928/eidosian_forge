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
def test_svds_input_validation_maxiter_2(self):
    message = 'int() argument must be a'
    with pytest.raises(TypeError, match=re.escape(message)):
        svds(np.eye(10), maxiter=[], solver=self.solver)
    message = 'invalid literal for int()'
    with pytest.raises(ValueError, match=message):
        svds(np.eye(10), maxiter='hi', solver=self.solver)