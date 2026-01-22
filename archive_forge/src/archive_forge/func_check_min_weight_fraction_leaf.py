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
def check_min_weight_fraction_leaf(name, datasets, sparse_container=None):
    """Test if leaves contain at least min_weight_fraction_leaf of the
    training set"""
    X = DATASETS[datasets]['X'].astype(np.float32)
    if sparse_container is not None:
        X = sparse_container(X)
    y = DATASETS[datasets]['y']
    weights = rng.rand(X.shape[0])
    total_weight = np.sum(weights)
    TreeEstimator = ALL_TREES[name]
    for max_leaf_nodes, frac in product((None, 1000), np.linspace(0, 0.5, 6)):
        est = TreeEstimator(min_weight_fraction_leaf=frac, max_leaf_nodes=max_leaf_nodes, random_state=0)
        est.fit(X, y, sample_weight=weights)
        if sparse_container is not None:
            out = est.tree_.apply(X.tocsr())
        else:
            out = est.tree_.apply(X)
        node_weights = np.bincount(out, weights=weights)
        leaf_weights = node_weights[node_weights != 0]
        assert np.min(leaf_weights) >= total_weight * est.min_weight_fraction_leaf, 'Failed with {0} min_weight_fraction_leaf={1}'.format(name, est.min_weight_fraction_leaf)
    total_weight = X.shape[0]
    for max_leaf_nodes, frac in product((None, 1000), np.linspace(0, 0.5, 6)):
        est = TreeEstimator(min_weight_fraction_leaf=frac, max_leaf_nodes=max_leaf_nodes, random_state=0)
        est.fit(X, y)
        if sparse_container is not None:
            out = est.tree_.apply(X.tocsr())
        else:
            out = est.tree_.apply(X)
        node_weights = np.bincount(out)
        leaf_weights = node_weights[node_weights != 0]
        assert np.min(leaf_weights) >= total_weight * est.min_weight_fraction_leaf, 'Failed with {0} min_weight_fraction_leaf={1}'.format(name, est.min_weight_fraction_leaf)