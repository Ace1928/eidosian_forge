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
def test_ar_covsolve():
    np.random.seed(123)
    c = cov_struct.Autoregressive(grid=True)
    c.dep_params = 0.4
    for d in (1, 2, 4):
        for q in (1, 4):
            ii = np.arange(d)
            mat = 0.4 ** np.abs(np.subtract.outer(ii, ii))
            sd = np.random.uniform(size=d)
            if q == 1:
                z = np.random.normal(size=d)
            else:
                z = np.random.normal(size=(d, q))
            sm = np.diag(sd)
            z1 = np.linalg.solve(sm, np.linalg.solve(mat, np.linalg.solve(sm, z)))
            z2 = c.covariance_matrix_solve(np.zeros_like(sd), np.zeros_like(sd), sd, [z])
            assert_allclose(z1, z2[0], rtol=1e-05, atol=1e-05)