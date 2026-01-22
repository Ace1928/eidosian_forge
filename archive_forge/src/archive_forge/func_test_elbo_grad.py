import numpy as np
from statsmodels.genmod.bayes_mixed_glm import (BinomialBayesMixedGLM,
import pandas as pd
from scipy import sparse
from numpy.testing import assert_allclose, assert_equal
from scipy.optimize import approx_fprime
def test_elbo_grad():
    for f in range(2):
        for j in range(2):
            if f == 0:
                if j == 0:
                    y, exog_fe, exog_vc, ident = gen_simple_logit(10, 10, 2)
                else:
                    y, exog_fe, exog_vc, ident = gen_crossed_logit(10, 10, 1, 2)
            elif f == 1:
                if j == 0:
                    y, exog_fe, exog_vc, ident = gen_simple_poisson(10, 10, 0.5)
                else:
                    y, exog_fe, exog_vc, ident = gen_crossed_poisson(10, 10, 1, 0.5)
            exog_vc = sparse.csr_matrix(exog_vc)
            if f == 0:
                glmm1 = BinomialBayesMixedGLM(y, exog_fe, exog_vc, ident, vcp_p=0.5)
            else:
                glmm1 = PoissonBayesMixedGLM(y, exog_fe, exog_vc, ident, vcp_p=0.5)
            rslt1 = glmm1.fit_map()
            for k in range(3):
                if k == 0:
                    vb_mean = rslt1.params
                    vb_sd = np.ones_like(vb_mean)
                elif k == 1:
                    vb_mean = np.zeros(len(vb_mean))
                    vb_sd = np.ones_like(vb_mean)
                else:
                    vb_mean = np.random.normal(size=len(vb_mean))
                    vb_sd = np.random.uniform(1, 2, size=len(vb_mean))
                mean_grad, sd_grad = glmm1.vb_elbo_grad(vb_mean, vb_sd)

                def elbo(vec):
                    n = len(vec) // 2
                    return glmm1.vb_elbo(vec[:n], vec[n:])
                x = np.concatenate((vb_mean, vb_sd))
                g1 = approx_fprime(x, elbo, 1e-05)
                n = len(x) // 2
                mean_grad_n = g1[:n]
                sd_grad_n = g1[n:]
                assert_allclose(mean_grad, mean_grad_n, atol=0.01, rtol=0.01)
                assert_allclose(sd_grad, sd_grad_n, atol=0.01, rtol=0.01)