from collections import defaultdict
import numpy as np
from numpy import hstack, vstack
from numpy.linalg import inv, svd
import scipy
import scipy.stats
from statsmodels.iolib.summary import Summary
from statsmodels.iolib.table import SimpleTable
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.sm_exceptions import HypothesisTestWarning
from statsmodels.tools.validation import string_like
import statsmodels.tsa.base.tsa_model as tsbase
from statsmodels.tsa.coint_tables import c_sja, c_sjt
from statsmodels.tsa.tsatools import duplication_matrix, lagmat, vec
from statsmodels.tsa.vector_ar.hypothesis_test_results import (
import statsmodels.tsa.vector_ar.irf as irf
import statsmodels.tsa.vector_ar.plotting as plot
from statsmodels.tsa.vector_ar.util import get_index, seasonal_dummies
from statsmodels.tsa.vector_ar.var_model import (
def select_coint_rank(endog, det_order, k_ar_diff, method='trace', signif=0.05):
    """Calculate the cointegration rank of a VECM.

    Parameters
    ----------
    endog : array_like (nobs_tot x neqs)
        The data with presample.
    det_order : int
        * -1 - no deterministic terms
        * 0 - constant term
        * 1 - linear trend
    k_ar_diff : int, nonnegative
        Number of lagged differences in the model.
    method : str, {``"trace"``, ``"maxeig"``}, default: ``"trace"``
        If ``"trace"``, the trace test statistic is used. If ``"maxeig"``, the
        maximum eigenvalue test statistic is used.
    signif : float, {0.1, 0.05, 0.01}, default: 0.05
        The test's significance level.

    Returns
    -------
    rank : :class:`CointRankResults`
        A :class:`CointRankResults` object containing the cointegration rank suggested
        by the test and allowing a summary to be printed.
    """
    if method not in ['trace', 'maxeig']:
        raise ValueError("The method argument has to be either 'trace' or'maximum eigenvalue'.")
    if det_order not in [-1, 0, 1]:
        if type(det_order) is int and det_order > 1:
            raise ValueError('A det_order greather than 1 is not supported.Use a value of -1, 0, or 1.')
        else:
            raise ValueError('det_order must be -1, 0, or 1.')
    possible_signif_values = [0.1, 0.05, 0.01]
    if signif not in possible_signif_values:
        raise ValueError('Please choose a significance level from {0.1, 0.05,0.01}')
    coint_result = coint_johansen(endog, det_order, k_ar_diff)
    test_stat = coint_result.lr1 if method == 'trace' else coint_result.lr2
    crit_vals = coint_result.cvt if method == 'trace' else coint_result.cvm
    signif_index = possible_signif_values.index(signif)
    neqs = endog.shape[1]
    r_0 = 0
    while r_0 < neqs:
        if test_stat[r_0] < crit_vals[r_0, signif_index]:
            break
        else:
            r_0 += 1
    return CointRankResults(r_0, neqs, test_stat[:r_0 + 1], crit_vals[:r_0 + 1, signif_index], method, signif)