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
@pytest.mark.parametrize('Loss', (HalfSquaredError, AbsoluteError))
def test_sum_hessians_are_sample_weight(Loss):
    rng = np.random.RandomState(0)
    n_samples = 1000
    n_features = 2
    X, y = make_regression(n_samples=n_samples, n_features=n_features, random_state=rng)
    bin_mapper = _BinMapper()
    X_binned = bin_mapper.fit_transform(X)
    sample_weight = rng.normal(size=n_samples)
    loss = Loss(sample_weight=sample_weight)
    gradients, hessians = loss.init_gradient_and_hessian(n_samples=n_samples, dtype=G_H_DTYPE)
    gradients, hessians = (gradients.reshape((-1, 1)), hessians.reshape((-1, 1)))
    raw_predictions = rng.normal(size=(n_samples, 1))
    loss.gradient_hessian(y_true=y, raw_prediction=raw_predictions, sample_weight=sample_weight, gradient_out=gradients, hessian_out=hessians, n_threads=n_threads)
    sum_sw = np.zeros(shape=(n_features, bin_mapper.n_bins))
    for feature_idx in range(n_features):
        for sample_idx in range(n_samples):
            sum_sw[feature_idx, X_binned[sample_idx, feature_idx]] += sample_weight[sample_idx]
    grower = TreeGrower(X_binned, gradients[:, 0], hessians[:, 0], n_bins=bin_mapper.n_bins)
    histograms = grower.histogram_builder.compute_histograms_brute(grower.root.sample_indices)
    for feature_idx in range(n_features):
        for bin_idx in range(bin_mapper.n_bins):
            assert histograms[feature_idx, bin_idx]['sum_hessians'] == pytest.approx(sum_sw[feature_idx, bin_idx], rel=1e-05)