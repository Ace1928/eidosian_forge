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
@pytest.mark.parametrize('scoring, validation_fraction, early_stopping, n_iter_no_change, tol', [('neg_mean_squared_error', 0.1, True, 5, 1e-07), ('neg_mean_squared_error', None, True, 5, 0.1), (None, 0.1, True, 5, 1e-07), (None, None, True, 5, 0.1), ('loss', 0.1, True, 5, 1e-07), ('loss', None, True, 5, 0.1), (None, None, False, 5, 0.0)])
def test_early_stopping_regression(scoring, validation_fraction, early_stopping, n_iter_no_change, tol):
    max_iter = 200
    X, y = make_regression(n_samples=50, random_state=0)
    gb = HistGradientBoostingRegressor(verbose=1, min_samples_leaf=5, scoring=scoring, tol=tol, early_stopping=early_stopping, validation_fraction=validation_fraction, max_iter=max_iter, n_iter_no_change=n_iter_no_change, random_state=0)
    gb.fit(X, y)
    if early_stopping:
        assert n_iter_no_change <= gb.n_iter_ < max_iter
    else:
        assert gb.n_iter_ == max_iter