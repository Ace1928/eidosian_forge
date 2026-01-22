import json
import os
import numpy as np
from numpy.testing import (
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest
from statsmodels.datasets import macrodata, sunspots
from statsmodels.regression.linear_model import OLS
import statsmodels.stats.diagnostic as smsdia
import statsmodels.stats.outliers_influence as oi
import statsmodels.stats.sandwich_covariance as sw
from statsmodels.tools.tools import Bunch, add_constant
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
def test_influence_wrapped():
    from pandas import DataFrame
    d = macrodata.load_pandas().data
    gs_l_realinv = 400 * np.log(d['realinv']).diff().dropna()
    gs_l_realgdp = 400 * np.log(d['realgdp']).diff().dropna()
    lint = d['realint'][:-1]
    gs_l_realgdp.index = lint.index
    gs_l_realinv.index = lint.index
    data = dict(const=np.ones_like(lint), lint=lint, lrealgdp=gs_l_realgdp)
    exog = DataFrame(data, columns=['const', 'lrealgdp', 'lint'])
    res = OLS(gs_l_realinv, exog).fit()
    infl = oi.OLSInfluence(res)
    df = infl.summary_frame()
    assert_(isinstance(df, DataFrame))
    path = os.path.join(cur_dir, 'results', 'influence_lsdiag_R.json')
    with open(path, encoding='utf-8') as fp:
        lsdiag = json.load(fp)
    c0, c1 = infl.cooks_distance
    assert_almost_equal(c0, lsdiag['cooks'], 12)
    assert_almost_equal(infl.hat_matrix_diag, lsdiag['hat'], 12)
    assert_almost_equal(infl.resid_studentized_internal, lsdiag['std.res'], 12)
    dffits, dffth = infl.dffits
    assert_almost_equal(dffits, lsdiag['dfits'], 12)
    assert_almost_equal(infl.resid_studentized_external, lsdiag['stud.res'], 12)
    fn = os.path.join(cur_dir, 'results/influence_measures_R.csv')
    infl_r = pd.read_csv(fn, index_col=0)
    infl_r2 = np.asarray(infl_r)
    assert_almost_equal(infl.dfbetas, infl_r2[:, :3], decimal=12)
    assert_almost_equal(infl.cov_ratio, infl_r2[:, 4], decimal=12)