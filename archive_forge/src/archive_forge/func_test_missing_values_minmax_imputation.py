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
def test_missing_values_minmax_imputation():

    class MinMaxImputer(TransformerMixin, BaseEstimator):

        def fit(self, X, y=None):
            mm = MinMaxScaler().fit(X)
            self.data_min_ = mm.data_min_
            self.data_max_ = mm.data_max_
            return self

        def transform(self, X):
            X_min, X_max = (X.copy(), X.copy())
            for feature_idx in range(X.shape[1]):
                nan_mask = np.isnan(X[:, feature_idx])
                X_min[nan_mask, feature_idx] = self.data_min_[feature_idx] - 1
                X_max[nan_mask, feature_idx] = self.data_max_[feature_idx] + 1
            return np.concatenate([X_min, X_max], axis=1)

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
    X_train, X_test, y_train, y_test = make_missing_value_data(n_samples=int(10000.0), seed=0)
    gbm1 = HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=5, random_state=0)
    gbm1.fit(X_train, y_train)
    gbm2 = make_pipeline(MinMaxImputer(), clone(gbm1))
    gbm2.fit(X_train, y_train)
    assert gbm1.score(X_train, y_train) == pytest.approx(gbm2.score(X_train, y_train))
    assert gbm1.score(X_test, y_test) == pytest.approx(gbm2.score(X_test, y_test))
    assert_allclose(gbm1.predict(X_train), gbm2.predict(X_train))
    assert_allclose(gbm1.predict(X_test), gbm2.predict(X_test))