from typing import NamedTuple
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
from pandas.testing import assert_index_equal
import pytest
from statsmodels.datasets import danish_data
from statsmodels.iolib.summary import Summary
from statsmodels.tools.sm_exceptions import SpecificationWarning
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.ardl.model import (
from statsmodels.tsa.deterministic import DeterministicProcess
@pytest.mark.parametrize('use_numpy', [True, False])
@pytest.mark.parametrize('use_t', [True, False])
def test_uecm_ci_repr(use_numpy, use_t):
    y = dane_data.lrm
    x = dane_data[['lry', 'ibo', 'ide']]
    if use_numpy:
        y = np.asarray(y)
        x = np.asarray(x)
    mod = UECM(y, 3, x, 3)
    res = mod.fit(use_t=use_t)
    if use_numpy:
        ci_params = res.params[:5].copy()
        ci_params /= ci_params[1]
    else:
        ci_params = res.params.iloc[:5].copy()
        ci_params /= ci_params['lrm.L1']
    assert_allclose(res.ci_params, ci_params)
    assert res.ci_bse.shape == (5,)
    assert res.ci_tvalues.shape == (5,)
    assert res.ci_pvalues.shape == (5,)
    assert 'Cointegrating Vector' in str(res.ci_summary())
    assert res.ci_conf_int().shape == (5, 2)
    assert res.ci_cov_params().shape == (5, 5)
    assert res.ci_resids.shape == dane_data.lrm.shape