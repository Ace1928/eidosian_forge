import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.utils import check_random_state
from sklearn.utils._testing import assert_allclose, assert_array_less
@pytest.mark.parametrize('algorithm, n_components', [('arpack', 55), ('arpack', 56), ('randomized', 56)])
def test_too_many_components(X_sparse, algorithm, n_components):
    tsvd = TruncatedSVD(n_components=n_components, algorithm=algorithm)
    with pytest.raises(ValueError):
        tsvd.fit(X_sparse)