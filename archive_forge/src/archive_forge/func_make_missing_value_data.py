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
def make_missing_value_data(n_samples=int(10000.0), seed=0):
    rng = np.random.RandomState(seed)
    X, y = make_regression(n_samples=n_samples, n_features=4, random_state=rng)
    X = KBinsDiscretizer(n_bins=42, encode='ordinal').fit_transform(X)
    rnd_mask = rng.rand(X.shape[0]) > 0.9
    X[rnd_mask, 0] = np.nan
    low_mask = X[:, 1] == 0
    X[low_mask, 1] = np.nan
    high_mask = X[:, 2] == X[:, 2].max()
    X[high_mask, 2] = np.nan
    y_max = np.percentile(y, 70)
    y_max_mask = y >= y_max
    y[y_max_mask] = y_max
    X[y_max_mask, 3] = np.nan
    for feature_idx in range(X.shape[1]):
        assert any(np.isnan(X[:, feature_idx]))
    return train_test_split(X, y, random_state=rng)