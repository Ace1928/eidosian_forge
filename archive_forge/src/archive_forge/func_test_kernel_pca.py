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
def test_kernel_pca():
    """Nominal test for all solvers and all known kernels + a custom one

    It tests
     - that fit_transform is equivalent to fit+transform
     - that the shapes of transforms and inverse transforms are correct
    """
    rng = np.random.RandomState(0)
    X_fit = rng.random_sample((5, 4))
    X_pred = rng.random_sample((2, 4))

    def histogram(x, y, **kwargs):
        assert kwargs == {}
        return np.minimum(x, y).sum()
    for eigen_solver in ('auto', 'dense', 'arpack', 'randomized'):
        for kernel in ('linear', 'rbf', 'poly', histogram):
            inv = not callable(kernel)
            kpca = KernelPCA(4, kernel=kernel, eigen_solver=eigen_solver, fit_inverse_transform=inv)
            X_fit_transformed = kpca.fit_transform(X_fit)
            X_fit_transformed2 = kpca.fit(X_fit).transform(X_fit)
            assert_array_almost_equal(np.abs(X_fit_transformed), np.abs(X_fit_transformed2))
            assert X_fit_transformed.size != 0
            X_pred_transformed = kpca.transform(X_pred)
            assert X_pred_transformed.shape[1] == X_fit_transformed.shape[1]
            if inv:
                X_pred2 = kpca.inverse_transform(X_pred_transformed)
                assert X_pred2.shape == X_pred.shape