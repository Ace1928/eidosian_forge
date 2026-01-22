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
@pytest.mark.parametrize('sparse_container', [None] + CSC_CONTAINERS)
def test_empty_leaf_infinite_threshold(sparse_container):
    data = np.random.RandomState(0).randn(100, 11) * 2e+38
    data = np.nan_to_num(data.astype('float32'))
    X = data[:, :-1]
    if sparse_container is not None:
        X = sparse_container(X)
    y = data[:, -1]
    tree = DecisionTreeRegressor(random_state=0).fit(X, y)
    terminal_regions = tree.apply(X)
    left_leaf = set(np.where(tree.tree_.children_left == TREE_LEAF)[0])
    empty_leaf = left_leaf.difference(terminal_regions)
    infinite_threshold = np.where(~np.isfinite(tree.tree_.threshold))[0]
    assert len(infinite_threshold) == 0
    assert len(empty_leaf) == 0