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
@pytest.mark.parametrize('HistGradientBoosting, X, y', [(HistGradientBoostingClassifier, X_classification, y_classification), (HistGradientBoostingRegressor, X_regression, y_regression), (HistGradientBoostingClassifier, X_multi_classification, y_multi_classification)])
def test_staged_predict(HistGradientBoosting, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
    gb = HistGradientBoosting(max_iter=10)
    with pytest.raises(NotFittedError):
        next(gb.staged_predict(X_test))
    gb.fit(X_train, y_train)
    method_names = ['predict'] if is_regressor(gb) else ['predict', 'predict_proba', 'decision_function']
    for method_name in method_names:
        staged_method = getattr(gb, 'staged_' + method_name)
        staged_predictions = list(staged_method(X_test))
        assert len(staged_predictions) == gb.n_iter_
        for n_iter, staged_predictions in enumerate(staged_method(X_test), 1):
            aux = HistGradientBoosting(max_iter=n_iter)
            aux.fit(X_train, y_train)
            pred_aux = getattr(aux, method_name)(X_test)
            assert_allclose(staged_predictions, pred_aux)
            assert staged_predictions.shape == pred_aux.shape