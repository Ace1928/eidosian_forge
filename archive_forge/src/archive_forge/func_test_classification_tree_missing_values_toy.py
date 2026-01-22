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
def test_classification_tree_missing_values_toy():
    """Check that we properly handle missing values in clasification trees using a toy
    dataset.

    The test is more involved because we use a case where we detected a regression
    in a random forest. We therefore define the seed and bootstrap indices to detect
    one of the non-frequent regression.

    Here, we check that the impurity is null or positive in the leaves.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/28254
    """
    X, y = datasets.load_iris(return_X_y=True)
    rng = np.random.RandomState(42)
    X_missing = X.copy()
    mask = rng.binomial(n=np.ones(shape=(1, 4), dtype=np.int32), p=X[:, [2]] / 8).astype(bool)
    X_missing[mask] = np.nan
    X_train, _, y_train, _ = train_test_split(X_missing, y, random_state=13)
    indices = np.array([2, 81, 39, 97, 91, 38, 46, 31, 101, 13, 89, 82, 100, 42, 69, 27, 81, 16, 73, 74, 51, 47, 107, 17, 75, 110, 20, 15, 104, 57, 26, 15, 75, 79, 35, 77, 90, 51, 46, 13, 94, 91, 23, 8, 93, 93, 73, 77, 12, 13, 74, 109, 110, 24, 10, 23, 104, 27, 92, 52, 20, 109, 8, 8, 28, 27, 35, 12, 12, 7, 43, 0, 30, 31, 78, 12, 24, 105, 50, 0, 73, 12, 102, 105, 13, 31, 1, 69, 11, 32, 75, 90, 106, 94, 60, 56, 35, 17, 62, 85, 81, 39, 80, 16, 63, 6, 80, 84, 3, 3, 76, 78], dtype=np.int32)
    tree = DecisionTreeClassifier(max_depth=3, max_features='sqrt', random_state=1857819720)
    tree.fit(X_train[indices], y_train[indices])
    assert all(tree.tree_.impurity >= 0)
    leaves_idx = np.flatnonzero((tree.tree_.children_left == -1) & (tree.tree_.n_node_samples == 1))
    assert_allclose(tree.tree_.impurity[leaves_idx], 0.0)