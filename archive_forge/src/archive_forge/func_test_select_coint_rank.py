import numpy as np
from numpy.testing import (
import pytest
import statsmodels.datasets.interest_inflation.data as e6
from statsmodels.tools.testing import assert_equal
from statsmodels.tsa.vector_ar.tests.JMulTi_results.parse_jmulti_vecm_output import (
from statsmodels.tsa.vector_ar.util import seasonal_dummies
from statsmodels.tsa.vector_ar.var_model import VARProcess
from statsmodels.tsa.vector_ar.vecm import (
def test_select_coint_rank():
    if debug_mode:
        if 'select_coint_rank' not in to_test:
            return
        else:
            print('\n\nSELECT_COINT_RANK\n', end='')
    endog = data[datasets[0]]
    neqs = endog.shape[1]
    trace_result = select_coint_rank(endog, 0, 3, method='trace')
    rank = trace_result.rank
    r_1 = trace_result.r_1
    test_stats = trace_result.test_stats
    crit_vals = trace_result.crit_vals
    if rank > 0:
        assert_equal(r_1[0], r_1[1])
        for i in range(rank):
            assert_(test_stats[i] > crit_vals[i])
    if rank < neqs:
        assert_(test_stats[rank] < crit_vals[rank])
    maxeig_result = select_coint_rank(endog, 0, 3, method='maxeig', signif=0.1)
    rank = maxeig_result.rank
    r_1 = maxeig_result.r_1
    test_stats = maxeig_result.test_stats
    crit_vals = maxeig_result.crit_vals
    if maxeig_result.rank > 0:
        assert_equal(r_1[0], r_1[1] - 1)
        for i in range(rank):
            assert_(test_stats[i] > crit_vals[i])
    if rank < neqs:
        assert_(test_stats[rank] < crit_vals[rank])