import copyreg
import io
import pickle
import re
import warnings
from unittest.mock import Mock
import joblib
import numpy as np
import pytest
from joblib.numpy_pickle import NumpyPickler
from numpy.testing import assert_allclose, assert_array_equal
import sklearn
from sklearn._loss.loss import (
from sklearn.base import BaseEstimator, TransformerMixin, clone, is_regressor
from sklearn.compose import make_column_transformer
from sklearn.datasets import make_classification, make_low_rank_matrix, make_regression
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
from sklearn.ensemble._hist_gradient_boosting.binning import _BinMapper
from sklearn.ensemble._hist_gradient_boosting.common import G_H_DTYPE
from sklearn.ensemble._hist_gradient_boosting.grower import TreeGrower
from sklearn.ensemble._hist_gradient_boosting.predictor import TreePredictor
from sklearn.exceptions import NotFittedError
from sklearn.metrics import get_scorer, mean_gamma_deviance, mean_poisson_deviance
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import KBinsDiscretizer, MinMaxScaler, OneHotEncoder
from sklearn.utils import _IS_32BIT, shuffle
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
from sklearn.utils._testing import _convert_container
@pytest.mark.parametrize('problem', ('regression', 'binary_classification', 'multiclass_classification'))
@pytest.mark.parametrize('duplication', ('half', 'all'))
def test_sample_weight_effect(problem, duplication):
    n_samples = 255
    n_features = 2
    if problem == 'regression':
        X, y = make_regression(n_samples=n_samples, n_features=n_features, n_informative=n_features, random_state=0)
        Klass = HistGradientBoostingRegressor
    else:
        n_classes = 2 if problem == 'binary_classification' else 3
        X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_features, n_redundant=0, n_clusters_per_class=1, n_classes=n_classes, random_state=0)
        Klass = HistGradientBoostingClassifier
    est = Klass(min_samples_leaf=1)
    if duplication == 'half':
        lim = n_samples // 2
    else:
        lim = n_samples
    X_dup = np.r_[X, X[:lim]]
    y_dup = np.r_[y, y[:lim]]
    sample_weight = np.ones(shape=n_samples)
    sample_weight[:lim] = 2
    est_sw = clone(est).fit(X, y, sample_weight=sample_weight)
    est_dup = clone(est).fit(X_dup, y_dup)
    assert np.allclose(est_sw._raw_predict(X_dup), est_dup._raw_predict(X_dup))