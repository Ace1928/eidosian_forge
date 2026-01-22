import warnings
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn import datasets
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS, LIL_CONTAINERS
def test_incremental_pca_fit_overflow_error():
    rng = np.random.RandomState(0)
    A = rng.rand(500000, 2)
    ipca = IncrementalPCA(n_components=2, batch_size=10000)
    ipca.fit(A)
    pca = PCA(n_components=2)
    pca.fit(A)
    np.testing.assert_allclose(ipca.singular_values_, pca.singular_values_)