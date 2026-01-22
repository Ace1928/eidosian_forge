import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import (
from sklearn.ensemble._hist_gradient_boosting.binning import _BinMapper
from sklearn.ensemble._hist_gradient_boosting.utils import get_equivalent_estimator
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
@pytest.mark.parametrize('seed', range(5))
@pytest.mark.parametrize('loss', ['squared_error', 'poisson', pytest.param('gamma', marks=pytest.mark.skip('LightGBM with gamma loss has larger deviation.'))])
@pytest.mark.parametrize('min_samples_leaf', (1, 20))
@pytest.mark.parametrize('n_samples, max_leaf_nodes', [(255, 4096), (1000, 8)])
def test_same_predictions_regression(seed, loss, min_samples_leaf, n_samples, max_leaf_nodes):
    pytest.importorskip('lightgbm')
    rng = np.random.RandomState(seed=seed)
    max_iter = 1
    max_bins = 255
    X, y = make_regression(n_samples=n_samples, n_features=5, n_informative=5, random_state=0)
    if loss in ('gamma', 'poisson'):
        y = np.abs(y) + np.mean(np.abs(y))
    if n_samples > 255:
        X = _BinMapper(n_bins=max_bins + 1).fit_transform(X).astype(np.float32)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=rng)
    est_sklearn = HistGradientBoostingRegressor(loss=loss, max_iter=max_iter, max_bins=max_bins, learning_rate=1, early_stopping=False, min_samples_leaf=min_samples_leaf, max_leaf_nodes=max_leaf_nodes)
    est_lightgbm = get_equivalent_estimator(est_sklearn, lib='lightgbm')
    est_lightgbm.set_params(min_sum_hessian_in_leaf=0)
    est_lightgbm.fit(X_train, y_train)
    est_sklearn.fit(X_train, y_train)
    X_train, X_test = (X_train.astype(np.float32), X_test.astype(np.float32))
    pred_lightgbm = est_lightgbm.predict(X_train)
    pred_sklearn = est_sklearn.predict(X_train)
    if loss in ('gamma', 'poisson'):
        assert np.mean(np.isclose(pred_lightgbm, pred_sklearn, rtol=0.01, atol=0.01)) > 0.65
    else:
        assert np.mean(np.isclose(pred_lightgbm, pred_sklearn, rtol=0.001)) > 1 - 0.01
    if max_leaf_nodes < 10 and n_samples >= 1000 and (loss in ('squared_error',)):
        pred_lightgbm = est_lightgbm.predict(X_test)
        pred_sklearn = est_sklearn.predict(X_test)
        assert np.mean(np.isclose(pred_lightgbm, pred_sklearn, rtol=0.0001)) > 1 - 0.01