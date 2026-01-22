import sys
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.decomposition import PCA, MiniBatchSparsePCA, SparsePCA
from sklearn.utils import check_random_state
from sklearn.utils._testing import (
def test_scaling_fit_transform():
    alpha = 1
    rng = np.random.RandomState(0)
    Y, _, _ = generate_toy_data(3, 1000, (8, 8), random_state=rng)
    spca_lars = SparsePCA(n_components=3, method='lars', alpha=alpha, random_state=rng)
    results_train = spca_lars.fit_transform(Y)
    results_test = spca_lars.transform(Y[:10])
    assert_allclose(results_train[0], results_test[0])