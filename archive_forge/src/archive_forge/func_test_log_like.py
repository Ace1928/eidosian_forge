import numpy as np
from numpy.testing import (
import pytest
import statsmodels.datasets.interest_inflation.data as e6
from statsmodels.tools.testing import assert_equal
from statsmodels.tsa.vector_ar.tests.JMulTi_results.parse_jmulti_vecm_output import (
from statsmodels.tsa.vector_ar.util import seasonal_dummies
from statsmodels.tsa.vector_ar.var_model import VARProcess
from statsmodels.tsa.vector_ar.vecm import (
def test_log_like():
    if debug_mode:
        if 'log_like' not in to_test:
            return
        else:
            print('\n\nLOG LIKELIHOOD', end='')
    for ds in datasets:
        for dt in ds.dt_s_list:
            if debug_mode:
                print('\n' + dt_s_tup_to_string(dt) + ': ', end='')
            exog = results_sm_exog[ds][dt].exog is not None
            exog_coint = results_sm_exog_coint[ds][dt].exog_coint is not None
            err_msg = build_err_msg(ds, dt, 'Log Likelihood')
            obtained = results_sm[ds][dt].llf
            obtained_exog = results_sm_exog[ds][dt].llf
            obtained_exog_coint = results_sm_exog_coint[ds][dt].llf
            nobs = results_sm[ds][dt].nobs
            desired = results_ref[ds][dt]['log_like'] * nobs / (nobs - 1)
            assert_allclose(obtained, desired, rtol, atol, False, err_msg)
            if exog:
                assert_equal(obtained_exog, obtained, 'WITH EXOG' + err_msg)
            if exog_coint:
                assert_equal(obtained_exog_coint, obtained, 'WITH EXOG_COINT' + err_msg)