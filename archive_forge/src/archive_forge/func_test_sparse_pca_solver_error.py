import re
import warnings
import numpy as np
import pytest
import scipy as sp
from numpy.testing import assert_array_equal
from sklearn import config_context, datasets
from sklearn.base import clone
from sklearn.datasets import load_iris, make_classification
from sklearn.decomposition import PCA
from sklearn.decomposition._pca import _assess_dimension, _infer_dimension
from sklearn.utils._array_api import (
from sklearn.utils._array_api import device as array_device
from sklearn.utils._testing import _array_api_for_tests, assert_allclose
from sklearn.utils.estimator_checks import (
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS
@pytest.mark.parametrize('svd_solver', ['randomized', 'full', 'auto'])
@pytest.mark.parametrize('sparse_container', CSR_CONTAINERS + CSC_CONTAINERS)
def test_sparse_pca_solver_error(global_random_seed, svd_solver, sparse_container):
    random_state = np.random.RandomState(global_random_seed)
    X = sparse_container(sp.sparse.random(SPARSE_M, SPARSE_N, random_state=random_state))
    pca = PCA(n_components=30, svd_solver=svd_solver)
    error_msg_pattern = f'PCA only support sparse inputs with the "arpack" solver, while "{svd_solver}" was passed'
    with pytest.raises(TypeError, match=error_msg_pattern):
        pca.fit(X)