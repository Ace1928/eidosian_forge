import warnings
import numpy as np
import pytest
import sklearn
from sklearn.datasets import load_iris, make_blobs, make_circles
from sklearn.decomposition import PCA, KernelPCA
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import Perceptron
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
from sklearn.utils.validation import _check_psd_eigenvalues
def test_kernel_pca_inverse_correct_gamma():
    """Check that gamma is set correctly when not provided.

    Non-regression test for #26280
    """
    rng = np.random.RandomState(0)
    X = rng.random_sample((5, 4))
    kwargs = {'n_components': 2, 'random_state': rng, 'fit_inverse_transform': True, 'kernel': 'rbf'}
    expected_gamma = 1 / X.shape[1]
    kpca1 = KernelPCA(gamma=None, **kwargs).fit(X)
    kpca2 = KernelPCA(gamma=expected_gamma, **kwargs).fit(X)
    assert kpca1.gamma_ == expected_gamma
    assert kpca2.gamma_ == expected_gamma
    X1_recon = kpca1.inverse_transform(kpca1.transform(X))
    X2_recon = kpca2.inverse_transform(kpca1.transform(X))
    assert_allclose(X1_recon, X2_recon)