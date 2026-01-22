from statsmodels.compat.python import lrange, lmap
import os
import copy
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pandas as pd
from statsmodels.tools.tools import add_constant
from statsmodels.regression.linear_model import OLS
import statsmodels.sandbox.regression.gmm as gmm
def test_bse_other(self):
    res1, res2 = (self.res1, self.res2)
    moms = res1.model.momcond(res1.params)
    w = res1.model.calc_weightmatrix(moms)
    bse = np.sqrt(np.diag(res1._cov_params(has_optimal_weights=False, weights=res1.weights)))
    bse = np.sqrt(np.diag(res1._cov_params(has_optimal_weights=False)))
    nobs = instrument.shape[0]
    w0inv = np.dot(instrument.T, instrument) / nobs
    q = self.res1.model.gmmobjective(self.res1.params, w)