import numpy as np
from statsmodels.genmod.bayes_mixed_glm import (BinomialBayesMixedGLM,
import pandas as pd
from scipy import sparse
from numpy.testing import assert_allclose, assert_equal
from scipy.optimize import approx_fprime
def test_crossed_logit_vb():
    y, exog_fe, exog_vc, ident = gen_crossed_logit(10, 10, 1, 2)
    glmm1 = BinomialBayesMixedGLM(y, exog_fe, exog_vc, ident, vcp_p=0.5, fe_p=0.5)
    rslt1 = glmm1.fit_map()
    glmm2 = BinomialBayesMixedGLM(y, exog_fe, exog_vc, ident, vcp_p=0.5, fe_p=0.5)
    rslt2 = glmm2.fit_vb(mean=rslt1.params)
    rslt1.summary()
    rslt2.summary()
    assert_allclose(rslt1.params[0:5], np.r_[-0.543073978, -2.46197518, -2.36582801, -0.00964030461, 0.00232701078], rtol=0.0001, atol=0.0001)
    assert_allclose(rslt1.cov_params().flat[0:5], np.r_[0.0412927123, -0.000204448923, 4.64829219e-05, 0.000120377543, -0.000145003234], rtol=0.0001, atol=0.0001)
    assert_allclose(rslt2.params[0:5], np.r_[-0.70834417, -0.3571011, 0.19126823, -0.36074489, 0.058976], rtol=0.0001, atol=0.0001)
    assert_allclose(rslt2.cov_params()[0:5], np.r_[0.05212492, 0.04729656, 0.03916944, 0.25921842, 0.25782576], rtol=0.0001, atol=0.0001)
    for rslt in (rslt1, rslt2):
        cp = rslt.cov_params()
        p = len(rslt.params)
        if rslt is rslt1:
            assert_equal(cp.shape, np.r_[p, p])
            np.linalg.cholesky(cp)
        else:
            assert_equal(cp.shape, np.r_[p,])
            assert_equal(cp > 0, True * np.ones(p))