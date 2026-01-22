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
def assert_is_subtree(tree, subtree):
    assert tree.node_count >= subtree.node_count
    assert tree.max_depth >= subtree.max_depth
    tree_c_left = tree.children_left
    tree_c_right = tree.children_right
    subtree_c_left = subtree.children_left
    subtree_c_right = subtree.children_right
    stack = [(0, 0)]
    while stack:
        tree_node_idx, subtree_node_idx = stack.pop()
        assert_array_almost_equal(tree.value[tree_node_idx], subtree.value[subtree_node_idx])
        assert_almost_equal(tree.impurity[tree_node_idx], subtree.impurity[subtree_node_idx])
        assert_almost_equal(tree.n_node_samples[tree_node_idx], subtree.n_node_samples[subtree_node_idx])
        assert_almost_equal(tree.weighted_n_node_samples[tree_node_idx], subtree.weighted_n_node_samples[subtree_node_idx])
        if subtree_c_left[subtree_node_idx] == subtree_c_right[subtree_node_idx]:
            assert_almost_equal(TREE_UNDEFINED, subtree.threshold[subtree_node_idx])
        else:
            assert_almost_equal(tree.threshold[tree_node_idx], subtree.threshold[subtree_node_idx])
            stack.append((tree_c_left[tree_node_idx], subtree_c_left[subtree_node_idx]))
            stack.append((tree_c_right[tree_node_idx], subtree_c_right[subtree_node_idx]))