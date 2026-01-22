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
def test_kernel_pca_n_components():
    """Test that `n_components` is correctly taken into account for projections

    For all solvers this tests that the output has the correct shape depending
    on the selected number of components.
    """
    rng = np.random.RandomState(0)
    X_fit = rng.random_sample((5, 4))
    X_pred = rng.random_sample((2, 4))
    for eigen_solver in ('dense', 'arpack', 'randomized'):
        for c in [1, 2, 4]:
            kpca = KernelPCA(n_components=c, eigen_solver=eigen_solver)
            shape = kpca.fit(X_fit).transform(X_pred).shape
            assert shape == (2, c)