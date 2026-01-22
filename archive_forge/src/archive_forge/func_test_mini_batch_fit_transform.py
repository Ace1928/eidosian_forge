import sys
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.decomposition import PCA, MiniBatchSparsePCA, SparsePCA
from sklearn.utils import check_random_state
from sklearn.utils._testing import (
@pytest.mark.skipif(True, reason='skipping mini_batch_fit_transform.')
def test_mini_batch_fit_transform():
    alpha = 1
    rng = np.random.RandomState(0)
    Y, _, _ = generate_toy_data(3, 10, (8, 8), random_state=rng)
    spca_lars = MiniBatchSparsePCA(n_components=3, random_state=0, alpha=alpha).fit(Y)
    U1 = spca_lars.transform(Y)
    if sys.platform == 'win32':
        import joblib
        _mp = joblib.parallel.multiprocessing
        joblib.parallel.multiprocessing = None
        try:
            spca = MiniBatchSparsePCA(n_components=3, n_jobs=2, alpha=alpha, random_state=0)
            U2 = spca.fit(Y).transform(Y)
        finally:
            joblib.parallel.multiprocessing = _mp
    else:
        spca = MiniBatchSparsePCA(n_components=3, n_jobs=2, alpha=alpha, random_state=0)
        U2 = spca.fit(Y).transform(Y)
    assert not np.all(spca_lars.components_ == 0)
    assert_array_almost_equal(U1, U2)
    spca_lasso = MiniBatchSparsePCA(n_components=3, method='cd', alpha=alpha, random_state=0).fit(Y)
    assert_array_almost_equal(spca_lasso.components_, spca_lars.components_)