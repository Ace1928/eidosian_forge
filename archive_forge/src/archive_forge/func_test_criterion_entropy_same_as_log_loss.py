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
@pytest.mark.parametrize('Tree', [DecisionTreeClassifier, ExtraTreeClassifier])
@pytest.mark.parametrize('n_classes', [2, 4])
def test_criterion_entropy_same_as_log_loss(Tree, n_classes):
    """Test that criterion=entropy gives same as log_loss."""
    n_samples, n_features = (50, 5)
    X, y = datasets.make_classification(n_classes=n_classes, n_samples=n_samples, n_features=n_features, n_informative=n_features, n_redundant=0, random_state=42)
    tree_log_loss = Tree(criterion='log_loss', random_state=43).fit(X, y)
    tree_entropy = Tree(criterion='entropy', random_state=43).fit(X, y)
    assert_tree_equal(tree_log_loss.tree_, tree_entropy.tree_, f"{Tree!r} with criterion 'entropy' and 'log_loss' gave different trees.")
    assert_allclose(tree_log_loss.predict(X), tree_entropy.predict(X))