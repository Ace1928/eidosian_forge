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
@pytest.mark.parametrize('GradientBoosting, X, y', [(HistGradientBoostingClassifier, X_classification, y_classification), (HistGradientBoostingRegressor, X_regression, y_regression)])
@pytest.mark.parametrize('params, err_msg', [({'interaction_cst': [0, 1]}, 'Interaction constraints must be a sequence of tuples or lists'), ({'interaction_cst': [{0, 9999}]}, 'Interaction constraints must consist of integer indices in \\[0, n_features - 1\\] = \\[.*\\], specifying the position of features,'), ({'interaction_cst': [{-1, 0}]}, 'Interaction constraints must consist of integer indices in \\[0, n_features - 1\\] = \\[.*\\], specifying the position of features,'), ({'interaction_cst': [{0.5}]}, 'Interaction constraints must consist of integer indices in \\[0, n_features - 1\\] = \\[.*\\], specifying the position of features,')])
def test_init_parameters_validation(GradientBoosting, X, y, params, err_msg):
    with pytest.raises(ValueError, match=err_msg):
        GradientBoosting(**params).fit(X, y)