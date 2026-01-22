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
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_kernel_pca_sparse(csr_container):
    """Test that kPCA works on a sparse data input.

    Same test as ``test_kernel_pca except inverse_transform`` since it's not
    implemented for sparse matrices.
    """
    rng = np.random.RandomState(0)
    X_fit = csr_container(rng.random_sample((5, 4)))
    X_pred = csr_container(rng.random_sample((2, 4)))
    for eigen_solver in ('auto', 'arpack', 'randomized'):
        for kernel in ('linear', 'rbf', 'poly'):
            kpca = KernelPCA(4, kernel=kernel, eigen_solver=eigen_solver, fit_inverse_transform=False, random_state=0)
            X_fit_transformed = kpca.fit_transform(X_fit)
            X_fit_transformed2 = kpca.fit(X_fit).transform(X_fit)
            assert_array_almost_equal(np.abs(X_fit_transformed), np.abs(X_fit_transformed2))
            X_pred_transformed = kpca.transform(X_pred)
            assert X_pred_transformed.shape[1] == X_fit_transformed.shape[1]
            with pytest.raises(NotFittedError):
                kpca.inverse_transform(X_pred_transformed)