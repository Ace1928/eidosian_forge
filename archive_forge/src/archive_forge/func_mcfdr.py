from pystatsmodels mailinglist 20100524
from collections import namedtuple
from statsmodels.compat.python import lzip, lrange
import copy
import math
import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
from scipy import stats, interpolate
from statsmodels.iolib.table import SimpleTable
from statsmodels.stats.multitest import multipletests, _ecdf as ecdf, fdrcorrection as fdrcorrection0, fdrcorrection_twostage
from statsmodels.graphics import utils
from statsmodels.tools.sm_exceptions import ValueWarning
def mcfdr(nrepl=100, nobs=50, ntests=10, ntrue=6, mu=0.5, alpha=0.05, rho=0.0):
    """MonteCarlo to test fdrcorrection
    """
    nfalse = ntests - ntrue
    locs = np.array([0.0] * ntrue + [mu] * (ntests - ntrue))
    results = []
    for i in range(nrepl):
        rvs = locs + randmvn(rho, size=(nobs, ntests))
        tt, tpval = stats.ttest_1samp(rvs, 0)
        res = fdrcorrection_bak(np.abs(tpval), alpha=alpha, method='i')
        res0 = fdrcorrection0(np.abs(tpval), alpha=alpha)
        results.append([np.sum(res[:ntrue]), np.sum(res[ntrue:])] + [np.sum(res0[:ntrue]), np.sum(res0[ntrue:])] + res.tolist() + np.sort(tpval).tolist() + [np.sum(tpval[:ntrue] < alpha), np.sum(tpval[ntrue:] < alpha)] + [np.sum(tpval[:ntrue] < alpha / ntests), np.sum(tpval[ntrue:] < alpha / ntests)])
    return np.array(results)