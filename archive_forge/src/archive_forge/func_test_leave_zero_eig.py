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
def test_leave_zero_eig():
    """Non-regression test for issue #12141 (PR #12143)

    This test checks that fit().transform() returns the same result as
    fit_transform() in case of non-removed zero eigenvalue.
    """
    X_fit = np.array([[1, 1], [0, 0]])
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        with np.errstate(all='warn'):
            k = KernelPCA(n_components=2, remove_zero_eig=False, eigen_solver='dense')
            A = k.fit(X_fit).transform(X_fit)
            B = k.fit_transform(X_fit)
            assert_array_almost_equal(np.abs(A), np.abs(B))