import pickle
import re
import warnings
import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose
import sklearn
from sklearn import config_context, datasets
from sklearn.base import (
from sklearn.decomposition import PCA
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils._mocking import MockDataFrame
from sklearn.utils._set_output import _get_output_config
from sklearn.utils._testing import (
def test_clone_protocol():
    """Checks that clone works with `__sklearn_clone__` protocol."""

    class FrozenEstimator(BaseEstimator):

        def __init__(self, fitted_estimator):
            self.fitted_estimator = fitted_estimator

        def __getattr__(self, name):
            return getattr(self.fitted_estimator, name)

        def __sklearn_clone__(self):
            return self

        def fit(self, *args, **kwargs):
            return self

        def fit_transform(self, *args, **kwargs):
            return self.fitted_estimator.transform(*args, **kwargs)
    X = np.array([[-1, -1], [-2, -1], [-3, -2]])
    pca = PCA().fit(X)
    components = pca.components_
    frozen_pca = FrozenEstimator(pca)
    assert_allclose(frozen_pca.components_, components)
    assert_array_equal(frozen_pca.get_feature_names_out(), pca.get_feature_names_out())
    X_new = np.asarray([[-1, 2], [3, 4], [1, 2]])
    frozen_pca.fit(X_new)
    assert_allclose(frozen_pca.components_, components)
    frozen_pca.fit_transform(X_new)
    assert_allclose(frozen_pca.components_, components)
    clone_frozen_pca = clone(frozen_pca)
    assert clone_frozen_pca is frozen_pca
    assert_allclose(clone_frozen_pca.components_, components)