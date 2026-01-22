import warnings
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn import datasets
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS, LIL_CONTAINERS
def test_incremental_pca_partial_fit_float_division():
    rng = np.random.RandomState(0)
    A = rng.randn(5, 3) + 2
    B = rng.randn(7, 3) + 5
    pca = IncrementalPCA(n_components=2)
    pca.partial_fit(A)
    pca.n_samples_seen_ = float(pca.n_samples_seen_)
    pca.partial_fit(B)
    singular_vals_float_samples_seen = pca.singular_values_
    pca2 = IncrementalPCA(n_components=2)
    pca2.partial_fit(A)
    pca2.partial_fit(B)
    singular_vals_int_samples_seen = pca2.singular_values_
    np.testing.assert_allclose(singular_vals_float_samples_seen, singular_vals_int_samples_seen)