import warnings
import numpy as np
import pytest
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin, clone, is_regressor
from sklearn.cluster import KMeans
from sklearn.compose import make_column_transformer
from sklearn.datasets import load_iris, make_classification, make_regression
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
from sklearn.exceptions import NotFittedError
from sklearn.inspection import partial_dependence
from sklearn.inspection._partial_dependence import (
from sklearn.linear_model import LinearRegression, LogisticRegression, MultiTaskLasso
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree.tests.test_tree import assert_is_subtree
from sklearn.utils import _IS_32BIT
from sklearn.utils._testing import assert_allclose, assert_array_equal
from sklearn.utils.validation import check_random_state
def test_partial_dependence_sample_weight_size_error():
    """Check that we raise an error when the size of `sample_weight` is not
    consistent with `X` and `y`.
    """
    est = LogisticRegression()
    (X, y), n_targets = binary_classification_data
    sample_weight = np.ones_like(y)
    est.fit(X, y)
    with pytest.raises(ValueError, match='sample_weight.shape =='):
        partial_dependence(est, X, features=[0], sample_weight=sample_weight[1:], grid_resolution=10)