import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.regression.quantile_regression import QuantReg
from statsmodels.robust import norms
from statsmodels.robust.robust_linear_model import RLM
def test_quantreg(self):
    t_eps = 1e-06
    mod1 = quantreg('dens ~ temp + I(temp ** 2.0)', self.df)
    y = mod1.endog
    xx = mod1.exog
    for q in [0.25, 0.75]:
        res1 = mod1.fit(q=q)
        mq_norm = norms.MQuantileNorm(q, norms.HuberT(t=t_eps))
        mod_rlm = RLM(y, xx, M=mq_norm)
        res_rlm = mod_rlm.fit()
        assert_allclose(res_rlm.params, res1.params, rtol=0.0005)
        assert_allclose(res_rlm.fittedvalues, res1.fittedvalues, rtol=0.001)
    q = 0.5
    t_eps = 0.01
    mod1 = RLM(y, xx, M=norms.HuberT(t=t_eps))
    res1 = mod1.fit()
    mq_norm = norms.MQuantileNorm(q, norms.HuberT(t=t_eps))
    mod_rlm = RLM(y, xx, M=mq_norm)
    res_rlm = mod_rlm.fit()
    assert_allclose(res_rlm.params, res1.params, rtol=1e-10)
    assert_allclose(res_rlm.fittedvalues, res1.fittedvalues, rtol=1e-10)