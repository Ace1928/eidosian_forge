import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import linalg, optimize
from sklearn._loss.loss import (
from sklearn.datasets import make_low_rank_matrix
from sklearn.linear_model._linear_loss import LinearModelLoss
from sklearn.utils.extmath import squared_norm
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('base_loss', LOSSES)
@pytest.mark.parametrize('sample_weight', [None, 'range'])
@pytest.mark.parametrize('l2_reg_strength', [0, 1])
@pytest.mark.parametrize('X_container', CSR_CONTAINERS + [None])
def test_loss_gradients_hessp_intercept(base_loss, sample_weight, l2_reg_strength, X_container):
    """Test that loss and gradient handle intercept correctly."""
    loss = LinearModelLoss(base_loss=base_loss(), fit_intercept=False)
    loss_inter = LinearModelLoss(base_loss=base_loss(), fit_intercept=True)
    n_samples, n_features = (10, 5)
    X, y, coef = random_X_y_coef(linear_model_loss=loss, n_samples=n_samples, n_features=n_features, seed=42)
    X[:, -1] = 1
    X_inter = X[:, :-1]
    if X_container is not None:
        X = X_container(X)
    if sample_weight == 'range':
        sample_weight = np.linspace(1, y.shape[0], num=y.shape[0])
    l, g = loss.loss_gradient(coef, X, y, sample_weight=sample_weight, l2_reg_strength=l2_reg_strength)
    _, hessp = loss.gradient_hessian_product(coef, X, y, sample_weight=sample_weight, l2_reg_strength=l2_reg_strength)
    l_inter, g_inter = loss_inter.loss_gradient(coef, X_inter, y, sample_weight=sample_weight, l2_reg_strength=l2_reg_strength)
    _, hessp_inter = loss_inter.gradient_hessian_product(coef, X_inter, y, sample_weight=sample_weight, l2_reg_strength=l2_reg_strength)
    assert l == pytest.approx(l_inter + 0.5 * l2_reg_strength * squared_norm(coef.T[-1]))
    g_inter_corrected = g_inter
    g_inter_corrected.T[-1] += l2_reg_strength * coef.T[-1]
    assert_allclose(g, g_inter_corrected)
    s = np.random.RandomState(42).randn(*coef.shape)
    h = hessp(s)
    h_inter = hessp_inter(s)
    h_inter_corrected = h_inter
    h_inter_corrected.T[-1] += l2_reg_strength * s.T[-1]
    assert_allclose(h, h_inter_corrected)