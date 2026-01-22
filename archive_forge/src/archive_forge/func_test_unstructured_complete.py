from statsmodels.compat import lrange
import os
import numpy as np
import pytest
from numpy.testing import (assert_almost_equal, assert_equal, assert_allclose,
import statsmodels.genmod.generalized_estimating_equations as gee
import statsmodels.tools as tools
import statsmodels.regression.linear_model as lm
from statsmodels.genmod import families
from statsmodels.genmod import cov_struct
import statsmodels.discrete.discrete_model as discrete
import pandas as pd
from scipy.stats.distributions import norm
import warnings
def test_unstructured_complete():
    np.random.seed(43)
    ngrp = 400
    cov = np.asarray([[1, 0.7, 0.2], [0.7, 1, 0.5], [0.2, 0.5, 1]])
    covr = np.linalg.cholesky(cov)
    e = np.random.normal(size=(ngrp, 3))
    e = np.dot(e, covr.T)
    xmat = np.random.normal(size=(3 * ngrp, 3))
    par = np.r_[1, -2, 0.1]
    ey = np.dot(xmat, par)
    y = ey + e.ravel()
    g = np.kron(np.arange(ngrp), np.ones(3))
    t = np.kron(np.ones(ngrp), np.r_[0, 1, 2]).astype(int)
    m = gee.GEE(y, xmat, time=t, cov_struct=cov_struct.Unstructured(), groups=g)
    r = m.fit()
    assert_allclose(r.params, par, 0.05, 0.5)
    assert_allclose(m.cov_struct.dep_params, cov, 0.05, 0.5)