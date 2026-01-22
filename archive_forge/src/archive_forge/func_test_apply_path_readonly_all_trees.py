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
@pytest.mark.parametrize('name', ALL_TREES)
@pytest.mark.parametrize('splitter', ['best', 'random'])
@pytest.mark.parametrize('sparse_container', [None] + CSC_CONTAINERS + CSR_CONTAINERS)
def test_apply_path_readonly_all_trees(name, splitter, sparse_container):
    dataset = DATASETS['clf_small']
    X_small = dataset['X'].astype(tree._tree.DTYPE, copy=False)
    if sparse_container is None:
        X_readonly = create_memmap_backed_data(X_small)
    else:
        X_readonly = sparse_container(dataset['X'])
        X_readonly.data = np.array(X_readonly.data, dtype=tree._tree.DTYPE)
        X_readonly.data, X_readonly.indices, X_readonly.indptr = create_memmap_backed_data((X_readonly.data, X_readonly.indices, X_readonly.indptr))
    y_readonly = create_memmap_backed_data(np.array(y_small, dtype=tree._tree.DTYPE))
    est = ALL_TREES[name](splitter=splitter)
    est.fit(X_readonly, y_readonly)
    assert_array_equal(est.predict(X_readonly), est.predict(X_small))
    assert_array_equal(est.decision_path(X_readonly).todense(), est.decision_path(X_small).todense())