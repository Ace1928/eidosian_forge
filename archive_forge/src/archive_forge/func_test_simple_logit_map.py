import numpy as np
from statsmodels.genmod.bayes_mixed_glm import (BinomialBayesMixedGLM,
import pandas as pd
from scipy import sparse
from numpy.testing import assert_allclose, assert_equal
from scipy.optimize import approx_fprime
def test_simple_logit_map():
    y, exog_fe, exog_vc, ident = gen_simple_logit(10, 10, 2)
    exog_vc = sparse.csr_matrix(exog_vc)
    glmm = BinomialBayesMixedGLM(y, exog_fe, exog_vc, ident, vcp_p=0.5)
    rslt = glmm.fit_map()
    assert_allclose(glmm.logposterior_grad(rslt.params), np.zeros_like(rslt.params), atol=0.001)
    for linear in (False, True):
        for exog in (None, exog_fe):
            pr1 = rslt.predict(linear=linear, exog=exog)
            pr2 = glmm.predict(rslt.params, linear=linear, exog=exog)
            assert_allclose(pr1, pr2)
            if not linear:
                assert_equal(pr1.min() >= 0, True)
                assert_equal(pr1.max() <= 1, True)