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
def test_importances_gini_equal_squared_error():
    X, y = datasets.make_classification(n_samples=2000, n_features=10, n_informative=3, n_redundant=0, n_repeated=0, shuffle=False, random_state=0)
    clf = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=0).fit(X, y)
    reg = DecisionTreeRegressor(criterion='squared_error', max_depth=5, random_state=0).fit(X, y)
    assert_almost_equal(clf.feature_importances_, reg.feature_importances_)
    assert_array_equal(clf.tree_.feature, reg.tree_.feature)
    assert_array_equal(clf.tree_.children_left, reg.tree_.children_left)
    assert_array_equal(clf.tree_.children_right, reg.tree_.children_right)
    assert_array_equal(clf.tree_.n_node_samples, reg.tree_.n_node_samples)