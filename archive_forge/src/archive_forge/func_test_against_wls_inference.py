from io import BytesIO
from itertools import product
import warnings
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from statsmodels import tools
from statsmodels.regression.linear_model import WLS
from statsmodels.regression.rolling import RollingWLS, RollingOLS
@pytest.mark.parametrize('cov_type', ['nonrobust', 'HC0'])
@pytest.mark.parametrize('use_t', [None, True, False])
def test_against_wls_inference(data, use_t, cov_type):
    y, x, w = data
    mod = RollingWLS(y, x, window=100, weights=w)
    res = mod.fit(use_t=use_t, cov_type=cov_type)
    ci = res.conf_int()
    res.cov_params()
    for i in range(100, y.shape[0]):
        _y = get_sub(y, i, 100)
        _x = get_sub(x, i, 100)
        wls = WLS(_y, _x, missing='drop').fit(use_t=use_t, cov_type=cov_type)
        assert_allclose(get_single(res.tvalues, i - 1), wls.tvalues)
        assert_allclose(get_single(res.bse, i - 1), wls.bse)
        assert_allclose(get_single(res.pvalues, i - 1), wls.pvalues, atol=1e-08)
        assert_allclose(get_single(res.fvalue, i - 1), wls.fvalue)
        with np.errstate(invalid='ignore'):
            assert_allclose(get_single(res.f_pvalue, i - 1), wls.f_pvalue, atol=1e-08)
        assert res.cov_type == wls.cov_type
        assert res.use_t == wls.use_t
        wls_ci = wls.conf_int()
        if isinstance(ci, pd.DataFrame):
            ci_val = ci.iloc[i - 1]
            ci_val = np.asarray(ci_val).reshape((-1, 2))
        else:
            ci_val = ci[i - 1].T
        assert_allclose(ci_val, wls_ci)