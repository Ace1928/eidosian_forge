import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pandas as pd
from statsmodels.regression.linear_model import OLS, WLS
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from statsmodels.regression._prediction import get_prediction
def test_predict_se():
    nsample = 50
    x1 = np.linspace(0, 20, nsample)
    x = np.c_[x1, (x1 - 5) ** 2, np.ones(nsample)]
    np.random.seed(0)
    beta = [0.5, -0.01, 5.0]
    y_true2 = np.dot(x, beta)
    w = np.ones(nsample)
    w[int(nsample * 6.0 / 10):] = 3
    sig = 0.5
    y2 = y_true2 + sig * w * np.random.normal(size=nsample)
    x2 = x[:, [0, 2]]
    res2 = OLS(y2, x2).fit()
    covb = res2.cov_params()
    predvar = res2.mse_resid + (x2 * np.dot(covb, x2.T).T).sum(1)
    predstd = np.sqrt(predvar)
    prstd, iv_l, iv_u = wls_prediction_std(res2)
    np.testing.assert_almost_equal(prstd, predstd, 15)
    q = 2.010634754696446
    ci_half = q * predstd
    np.testing.assert_allclose(iv_u, res2.fittedvalues + ci_half, rtol=1e-08)
    np.testing.assert_allclose(iv_l, res2.fittedvalues - ci_half, rtol=1e-08)
    prstd, iv_l, iv_u = wls_prediction_std(res2, x2[:3, :])
    np.testing.assert_equal(prstd, prstd[:3])
    np.testing.assert_allclose(iv_u, res2.fittedvalues[:3] + ci_half[:3], rtol=1e-08)
    np.testing.assert_allclose(iv_l, res2.fittedvalues[:3] - ci_half[:3], rtol=1e-08)
    res3 = WLS(y2, x2, 1.0 / w).fit()
    covb = res3.cov_params()
    predvar = res3.mse_resid * w + (x2 * np.dot(covb, x2.T).T).sum(1)
    predstd = np.sqrt(predvar)
    prstd, iv_l, iv_u = wls_prediction_std(res3)
    np.testing.assert_almost_equal(prstd, predstd, 15)
    q = 2.010634754696446
    ci_half = q * predstd
    np.testing.assert_allclose(iv_u, res3.fittedvalues + ci_half, rtol=1e-08)
    np.testing.assert_allclose(iv_l, res3.fittedvalues - ci_half, rtol=1e-08)
    prstd, iv_l, iv_u = wls_prediction_std(res3, x2[-1:, :], weights=3.0)
    np.testing.assert_equal(prstd, prstd[-1])
    prstd, iv_l, iv_u = wls_prediction_std(res3, x2[-1, :], weights=3.0)
    np.testing.assert_equal(prstd, prstd[-1])
    prstd, iv_l, iv_u = wls_prediction_std(res3, x2[-2:, :], weights=3.0)
    np.testing.assert_equal(prstd, prstd[-2:])
    prstd, iv_l, iv_u = wls_prediction_std(res3, x2[-2:, :], weights=[3, 3])
    np.testing.assert_equal(prstd, prstd[-2:])
    prstd, iv_l, iv_u = wls_prediction_std(res3, x2[:3, :])
    np.testing.assert_equal(prstd, prstd[:3])
    np.testing.assert_allclose(iv_u, res3.fittedvalues[:3] + ci_half[:3], rtol=1e-08)
    np.testing.assert_allclose(iv_l, res3.fittedvalues[:3] - ci_half[:3], rtol=1e-08)
    np.testing.assert_raises(ValueError, wls_prediction_std, res3, x2[-1, 0], weights=3.0)
    sew1 = wls_prediction_std(res3, x2[-3:, :])[0] ** 2
    for wv in np.linspace(0.5, 3, 5):
        sew = wls_prediction_std(res3, x2[-3:, :], weights=1.0 / wv)[0] ** 2
        np.testing.assert_allclose(sew, sew1 + res3.scale * (wv - 1))