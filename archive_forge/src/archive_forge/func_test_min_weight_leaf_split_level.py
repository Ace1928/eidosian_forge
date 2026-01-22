import copy
import copyreg
import io
import pickle
import struct
from itertools import chain, product
import joblib
import numpy as np
import pytest
from joblib.numpy_pickle import NumpyPickler
from numpy.testing import assert_allclose
from sklearn import clone, datasets, tree
from sklearn.dummy import DummyRegressor
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, mean_poisson_deviance, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.random_projection import _sparse_random_matrix
from sklearn.tree import (
from sklearn.tree._classes import (
from sklearn.tree._tree import (
from sklearn.tree._tree import Tree as CythonTree
from sklearn.utils import _IS_32BIT, compute_sample_weight
from sklearn.utils._testing import (
from sklearn.utils.estimator_checks import check_sample_weights_invariance
from sklearn.utils.fixes import COO_CONTAINERS, CSC_CONTAINERS, CSR_CONTAINERS
from sklearn.utils.validation import check_random_state
@pytest.mark.parametrize('name', ALL_TREES)
@pytest.mark.parametrize('sparse_container', [None] + CSC_CONTAINERS)
def test_min_weight_leaf_split_level(name, sparse_container):
    TreeEstimator = ALL_TREES[name]
    X = np.array([[0], [0], [0], [0], [1]])
    y = [0, 0, 0, 0, 1]
    sample_weight = [0.2, 0.2, 0.2, 0.2, 0.2]
    if sparse_container is not None:
        X = sparse_container(X)
    est = TreeEstimator(random_state=0)
    est.fit(X, y, sample_weight=sample_weight)
    assert est.tree_.max_depth == 1
    est = TreeEstimator(random_state=0, min_weight_fraction_leaf=0.4)
    est.fit(X, y, sample_weight=sample_weight)
    assert est.tree_.max_depth == 0