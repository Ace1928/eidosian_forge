import sys
from io import StringIO
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from scipy.linalg import block_diag
from scipy.special import psi
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition._online_lda_fast import (
from sklearn.exceptions import NotFittedError
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('learning_method', ('batch', 'online'))
def test_lda_dtype_match(learning_method, global_dtype):
    """Check data type preservation of fitted attributes."""
    rng = np.random.RandomState(0)
    X = rng.uniform(size=(20, 10)).astype(global_dtype, copy=False)
    lda = LatentDirichletAllocation(n_components=5, random_state=0, learning_method=learning_method)
    lda.fit(X)
    assert lda.components_.dtype == global_dtype
    assert lda.exp_dirichlet_component_.dtype == global_dtype