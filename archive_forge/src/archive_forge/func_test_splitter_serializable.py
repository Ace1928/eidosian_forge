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
@pytest.mark.parametrize('Splitter', chain(DENSE_SPLITTERS.values(), SPARSE_SPLITTERS.values()))
def test_splitter_serializable(Splitter):
    """Check that splitters are serializable."""
    rng = np.random.RandomState(42)
    max_features = 10
    n_outputs, n_classes = (2, np.array([3, 2], dtype=np.intp))
    criterion = CRITERIA_CLF['gini'](n_outputs, n_classes)
    splitter = Splitter(criterion, max_features, 5, 0.5, rng, monotonic_cst=None)
    splitter_serialize = pickle.dumps(splitter)
    splitter_back = pickle.loads(splitter_serialize)
    assert splitter_back.max_features == max_features
    assert isinstance(splitter_back, Splitter)