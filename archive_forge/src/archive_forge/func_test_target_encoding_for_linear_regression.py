import re
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import (
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
@pytest.mark.filterwarnings('ignore:In version 1.5 onwards, subsample=200_000')
@pytest.mark.parametrize('smooth', [0.0, 'auto'])
def test_target_encoding_for_linear_regression(smooth, global_random_seed):
    linear_regression = Ridge(alpha=1e-06, solver='lsqr', fit_intercept=False)
    n_samples = 50000
    rng = np.random.RandomState(global_random_seed)
    y = rng.randn(n_samples)
    noise = 0.8 * rng.randn(n_samples)
    n_categories = 100
    X_informative = KBinsDiscretizer(n_bins=n_categories, encode='ordinal', strategy='uniform', random_state=rng).fit_transform((y + noise).reshape(-1, 1))
    permutated_labels = rng.permutation(n_categories)
    X_informative = permutated_labels[X_informative.astype(np.int32)]
    X_shuffled = rng.permutation(X_informative)
    X_near_unique_categories = rng.choice(int(0.9 * n_samples), size=n_samples, replace=True).reshape(-1, 1)
    X = np.concatenate([X_informative, X_shuffled, X_near_unique_categories], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    raw_model = linear_regression.fit(X_train, y_train)
    assert raw_model.score(X_train, y_train) < 0.1
    assert raw_model.score(X_test, y_test) < 0.1
    model_with_cv = make_pipeline(TargetEncoder(smooth=smooth, random_state=rng), linear_regression).fit(X_train, y_train)
    coef = model_with_cv[-1].coef_
    assert model_with_cv.score(X_train, y_train) > 0.5, coef
    assert model_with_cv.score(X_test, y_test) > 0.5, coef
    assert coef[0] == pytest.approx(1, abs=0.01)
    assert (np.abs(coef[1:]) < 0.2).all()
    target_encoder = TargetEncoder(smooth=smooth, random_state=rng).fit(X_train, y_train)
    X_enc_no_cv_train = target_encoder.transform(X_train)
    X_enc_no_cv_test = target_encoder.transform(X_test)
    model_no_cv = linear_regression.fit(X_enc_no_cv_train, y_train)
    coef = model_no_cv.coef_
    assert model_no_cv.score(X_enc_no_cv_train, y_train) > 0.7, coef
    assert model_no_cv.score(X_enc_no_cv_test, y_test) < 0.5, coef
    assert abs(coef[0]) < abs(coef[2])