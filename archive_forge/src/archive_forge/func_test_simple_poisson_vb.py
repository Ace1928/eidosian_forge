import numpy as np
from statsmodels.genmod.bayes_mixed_glm import (BinomialBayesMixedGLM,
import pandas as pd
from scipy import sparse
from numpy.testing import assert_allclose, assert_equal
from scipy.optimize import approx_fprime
def test_simple_poisson_vb():
    y, exog_fe, exog_vc, ident = gen_simple_poisson(10, 10, 1)
    exog_vc = sparse.csr_matrix(exog_vc)
    glmm1 = PoissonBayesMixedGLM(y, exog_fe, exog_vc, ident, vcp_p=0.5)
    rslt1 = glmm1.fit_map()
    glmm2 = PoissonBayesMixedGLM(y, exog_fe, exog_vc, ident, vcp_p=0.5)
    rslt2 = glmm2.fit_vb(rslt1.params)
    rslt1.summary()
    rslt2.summary()
    assert_allclose(rslt1.params[0:5], np.r_[-0.07233493, -0.06706505, -0.47159649, 1.12575122, -1.02442201], rtol=0.0001, atol=0.0001)
    assert_allclose(rslt1.cov_params().flat[0:5], np.r_[0.00790914, 0.00080666, -0.00050719, 0.00022648, 0.00046235], rtol=0.0001, atol=0.0001)
    assert_allclose(rslt2.params[0:5], np.r_[-0.07088814, -0.06373107, -0.22770786, 1.12923746, -1.26161339], rtol=0.0001, atol=0.0001)
    assert_allclose(rslt2.cov_params()[0:5], np.r_[0.00747782, 0.0092554, 0.04508904, 0.02934488, 0.20312746], rtol=0.0001, atol=0.0001)
    for rslt in (rslt1, rslt2):
        cp = rslt.cov_params()
        p = len(rslt.params)
        if rslt is rslt1:
            assert_equal(cp.shape, np.r_[p, p])
            np.linalg.cholesky(cp)
        else:
            assert_equal(cp.shape, np.r_[p,])
            assert_equal(cp > 0, True * np.ones(p))