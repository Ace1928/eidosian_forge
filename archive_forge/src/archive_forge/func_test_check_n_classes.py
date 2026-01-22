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
def test_check_n_classes():
    expected_dtype = np.dtype(np.int32) if _IS_32BIT else np.dtype(np.int64)
    allowed_dtypes = [np.dtype(np.int32), np.dtype(np.int64)]
    allowed_dtypes += [dt.newbyteorder() for dt in allowed_dtypes]
    n_classes = np.array([0, 1], dtype=expected_dtype)
    for dt in allowed_dtypes:
        _check_n_classes(n_classes.astype(dt), expected_dtype)
    with pytest.raises(ValueError, match='Wrong dimensions.+n_classes'):
        wrong_dim_n_classes = np.array([[0, 1]], dtype=expected_dtype)
        _check_n_classes(wrong_dim_n_classes, expected_dtype)
    with pytest.raises(ValueError, match='n_classes.+incompatible dtype'):
        wrong_dtype_n_classes = n_classes.astype(np.float64)
        _check_n_classes(wrong_dtype_n_classes, expected_dtype)