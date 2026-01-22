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
@pytest.mark.parametrize('sparse_container', [None] + CSR_CONTAINERS)
@pytest.mark.parametrize('tree', [DecisionTreeClassifier(splitter='random'), DecisionTreeRegressor(criterion='absolute_error')])
def test_missing_value_errors(sparse_container, tree):
    """Check unsupported configurations for missing values."""
    X = np.array([[1, 2, 3, 5, np.nan, 10, 20, 30, 60, np.nan]]).T
    y = np.array([0] * 5 + [1] * 5)
    if sparse_container is not None:
        X = sparse_container(X)
    with pytest.raises(ValueError, match='Input X contains NaN'):
        tree.fit(X, y)