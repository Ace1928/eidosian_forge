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
@pytest.mark.parametrize('Tree', ALL_TREES.values())
def test_min_sample_split_1_error(Tree):
    """Check that an error is raised when min_sample_split=1.

    non-regression test for issue gh-25481.
    """
    X = np.array([[0, 0], [1, 1]])
    y = np.array([0, 1])
    Tree(min_samples_split=1.0).fit(X, y)
    tree = Tree(min_samples_split=1)
    msg = "'min_samples_split' .* must be an int in the range \\[2, inf\\) or a float in the range \\(0.0, 1.0\\]"
    with pytest.raises(ValueError, match=msg):
        tree.fit(X, y)