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
@pytest.mark.parametrize('which', ('LA', 'SA', 'ekki', 0))
def test_svds_input_validation_which(self, which):
    with pytest.raises(ValueError, match='`which` must be in'):
        svds(np.eye(10), which=which, solver=self.solver)