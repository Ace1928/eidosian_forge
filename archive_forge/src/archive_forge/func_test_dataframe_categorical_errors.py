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
@pytest.mark.parametrize('dataframe_lib', ['pandas', 'polars'])
@pytest.mark.parametrize('HistGradientBoosting', [HistGradientBoostingClassifier, HistGradientBoostingRegressor])
def test_dataframe_categorical_errors(dataframe_lib, HistGradientBoosting):
    """Check error cases for pandas categorical feature."""
    pytest.importorskip(dataframe_lib)
    msg = "Categorical feature 'f_cat' is expected to have a cardinality <= 16"
    hist = HistGradientBoosting(categorical_features='from_dtype', max_bins=16)
    rng = np.random.RandomState(42)
    f_cat = rng.randint(0, high=100, size=100).astype(str)
    X_df = _convert_container(f_cat[:, None], dataframe_lib, ['f_cat'], categorical_feature_names=['f_cat'])
    y = rng.randint(0, high=2, size=100)
    with pytest.raises(ValueError, match=msg):
        hist.fit(X_df, y)