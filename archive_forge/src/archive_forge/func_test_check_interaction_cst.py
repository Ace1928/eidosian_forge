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
@pytest.mark.parametrize('interaction_cst, n_features, result', [(None, 931, None), ([{0, 1}], 2, [{0, 1}]), ('pairwise', 2, [{0, 1}]), ('pairwise', 4, [{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}]), ('no_interactions', 2, [{0}, {1}]), ('no_interactions', 4, [{0}, {1}, {2}, {3}]), ([(1, 0), [5, 1]], 6, [{0, 1}, {1, 5}, {2, 3, 4}])])
def test_check_interaction_cst(interaction_cst, n_features, result):
    """Check that _check_interaction_cst returns the expected list of sets"""
    est = HistGradientBoostingRegressor()
    est.set_params(interaction_cst=interaction_cst)
    assert est._check_interaction_cst(n_features) == result