import numpy as np
from statsmodels.genmod.bayes_mixed_glm import (BinomialBayesMixedGLM,
import pandas as pd
from scipy import sparse
from numpy.testing import assert_allclose, assert_equal
from scipy.optimize import approx_fprime
def test_crossed_poisson_map():
    y, exog_fe, exog_vc, ident = gen_crossed_poisson(10, 10, 1, 1)
    exog_vc = sparse.csr_matrix(exog_vc)
    glmm = PoissonBayesMixedGLM(y, exog_fe, exog_vc, ident, vcp_p=0.5)
    rslt = glmm.fit_map()
    assert_allclose(glmm.logposterior_grad(rslt.params), np.zeros_like(rslt.params), atol=0.0001)
    cp = rslt.cov_params()
    p = len(rslt.params)
    assert_equal(cp.shape, np.r_[p, p])
    np.linalg.cholesky(cp)