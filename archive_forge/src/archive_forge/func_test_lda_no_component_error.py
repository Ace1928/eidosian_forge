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
def test_lda_no_component_error():
    rng = np.random.RandomState(0)
    X = rng.randint(4, size=(20, 10))
    lda = LatentDirichletAllocation()
    regex = "This LatentDirichletAllocation instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator."
    with pytest.raises(NotFittedError, match=regex):
        lda.perplexity(X)