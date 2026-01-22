import numpy as np
from statsmodels.genmod.bayes_mixed_glm import (BinomialBayesMixedGLM,
import pandas as pd
from scipy import sparse
from numpy.testing import assert_allclose, assert_equal
from scipy.optimize import approx_fprime
def test_scale_map():
    y, exog_fe, exog_vc, ident = gen_simple_logit(10, 10, 0)
    exog_fe -= exog_fe.mean(0)
    exog_fe /= exog_fe.std(0)
    exog_vc = sparse.csr_matrix(exog_vc)
    rslts = []
    for scale_fe in (False, True):
        glmm = BinomialBayesMixedGLM(y, exog_fe, exog_vc, ident, vcp_p=0.5, fe_p=0.5)
        rslt = glmm.fit_map(scale_fe=scale_fe)
        rslts.append(rslt)
    assert_allclose(rslts[0].params, rslts[1].params, rtol=0.0001)