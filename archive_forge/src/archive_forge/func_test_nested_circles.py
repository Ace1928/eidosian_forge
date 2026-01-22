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
def test_nested_circles():
    """Check that kPCA projects in a space where nested circles are separable

    Tests that 2D nested circles become separable with a perceptron when
    projected in the first 2 kPCA using an RBF kernel, while raw samples
    are not directly separable in the original space.
    """
    X, y = make_circles(n_samples=400, factor=0.3, noise=0.05, random_state=0)
    train_score = Perceptron(max_iter=5).fit(X, y).score(X, y)
    assert train_score < 0.8
    kpca = KernelPCA(kernel='rbf', n_components=2, fit_inverse_transform=True, gamma=2.0)
    X_kpca = kpca.fit_transform(X)
    train_score = Perceptron(max_iter=5).fit(X_kpca, y).score(X_kpca, y)
    assert train_score == 1.0