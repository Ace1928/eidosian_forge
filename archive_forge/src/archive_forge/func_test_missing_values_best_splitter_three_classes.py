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
@pytest.mark.parametrize('criterion', ['entropy', 'gini'])
def test_missing_values_best_splitter_three_classes(criterion):
    """Test when missing values are uniquely present in a class among 3 classes."""
    missing_values_class = 0
    X = np.array([[np.nan] * 4 + [0, 1, 2, 3, 8, 9, 11, 12]]).T
    y = np.array([missing_values_class] * 4 + [1] * 4 + [2] * 4)
    dtc = DecisionTreeClassifier(random_state=42, max_depth=2, criterion=criterion)
    dtc.fit(X, y)
    X_test = np.array([[np.nan, 3, 12]]).T
    y_nan_pred = dtc.predict(X_test)
    assert_array_equal(y_nan_pred, [missing_values_class, 1, 2])