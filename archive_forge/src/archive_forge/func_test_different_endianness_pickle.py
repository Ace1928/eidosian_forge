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
def test_different_endianness_pickle():
    X, y = datasets.make_classification(random_state=0)
    clf = DecisionTreeClassifier(random_state=0, max_depth=3)
    clf.fit(X, y)
    score = clf.score(X, y)

    def reduce_ndarray(arr):
        return arr.byteswap().view(arr.dtype.newbyteorder()).__reduce__()

    def get_pickle_non_native_endianness():
        f = io.BytesIO()
        p = pickle.Pickler(f)
        p.dispatch_table = copyreg.dispatch_table.copy()
        p.dispatch_table[np.ndarray] = reduce_ndarray
        p.dump(clf)
        f.seek(0)
        return f
    new_clf = pickle.load(get_pickle_non_native_endianness())
    new_score = new_clf.score(X, y)
    assert np.isclose(score, new_score)