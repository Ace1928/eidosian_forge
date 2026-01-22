import os
import numpy as np
from statsmodels.duration.survfunc import (
from numpy.testing import assert_allclose
import pandas as pd
import pytest
def test_bmt():
    cb = {'linear': [107, 276], 'cloglog': [86, 230], 'log': [107, 332], 'asinsqrt': [104, 276], 'logit': [104, 230]}
    dfa = bmt[bmt.Group == 'ALL']
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    fp = os.path.join(cur_dir, 'results', 'bmt_results.csv')
    rslt = pd.read_csv(fp)
    sf = SurvfuncRight(dfa['T'].values, dfa.Status.values)
    assert_allclose(sf.surv_times, rslt.t)
    assert_allclose(sf.surv_prob, rslt.s, atol=0.0001, rtol=0.0001)
    assert_allclose(sf.surv_prob_se, rslt.se, atol=0.0001, rtol=0.0001)
    for method in ('linear', 'cloglog', 'log', 'logit', 'asinsqrt'):
        lcb, ucb = sf.quantile_ci(0.25, method=method)
        assert_allclose(cb[method], np.r_[lcb, ucb])