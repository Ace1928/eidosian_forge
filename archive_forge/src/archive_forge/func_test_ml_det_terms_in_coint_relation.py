import numpy as np
from numpy.testing import (
import pytest
import statsmodels.datasets.interest_inflation.data as e6
from statsmodels.tools.testing import assert_equal
from statsmodels.tsa.vector_ar.tests.JMulTi_results.parse_jmulti_vecm_output import (
from statsmodels.tsa.vector_ar.util import seasonal_dummies
from statsmodels.tsa.vector_ar.var_model import VARProcess
from statsmodels.tsa.vector_ar.vecm import (
def test_ml_det_terms_in_coint_relation():
    if debug_mode:
        if 'det_coint' not in to_test:
            return
        print('\n\nDET_COEF_COINT', end='')
    for ds in datasets:
        for dt in ds.dt_s_list:
            if debug_mode:
                print('\n' + dt_s_tup_to_string(dt) + ': ', end='')
            exog = results_sm_exog[ds][dt].exog is not None
            exog_coint = results_sm_exog_coint[ds][dt].exog_coint is not None
            err_msg = build_err_msg(ds, dt, 'det terms in coint relation')
            dt_string = dt_s_tup_to_string(dt)
            obtained = results_sm[ds][dt].det_coef_coint
            obtained_exog = results_sm_exog[ds][dt].det_coef_coint
            obtained_exog_coint = results_sm_exog_coint[ds][dt].det_coef_coint
            if 'ci' not in dt_string and 'li' not in dt_string:
                if obtained.size > 0:
                    assert_(False, build_err_msg(ds, dt, 'There should not be any det terms in ' + 'cointegration for deterministic terms ' + dt_string))
                else:
                    assert_(True)
                continue
            desired = results_ref[ds][dt]['est']['det_coint']
            assert_allclose(obtained, desired, rtol, atol, False, err_msg)
            if exog:
                assert_equal(obtained_exog, obtained, 'WITH EXOG' + err_msg)
            if exog_coint:
                assert_equal(obtained_exog_coint, obtained, 'WITH EXOG_COINT' + err_msg)
            se_obtained = results_sm[ds][dt].stderr_det_coef_coint
            se_obtained_exog = results_sm_exog[ds][dt].stderr_det_coef_coint
            se_obtained_exog_coint = results_sm_exog_coint[ds][dt].stderr_det_coef_coint
            se_desired = results_ref[ds][dt]['se']['det_coint']
            assert_allclose(se_obtained, se_desired, rtol, atol, False, 'STANDARD ERRORS\n' + err_msg)
            if exog:
                assert_equal(se_obtained_exog, se_obtained, 'WITH EXOG' + err_msg)
            if exog_coint:
                assert_equal(se_obtained_exog_coint, se_obtained, 'WITH EXOG_COINT' + err_msg)
            t_obtained = results_sm[ds][dt].tvalues_det_coef_coint
            t_obtained_exog = results_sm_exog[ds][dt].tvalues_det_coef_coint
            t_obtained_exog_coint = results_sm_exog_coint[ds][dt].tvalues_det_coef_coint
            t_desired = results_ref[ds][dt]['t']['det_coint']
            assert_allclose(t_obtained, t_desired, rtol, atol, False, 't-VALUES\n' + err_msg)
            if exog:
                assert_equal(t_obtained_exog, t_obtained, 'WITH EXOG' + err_msg)
            if exog_coint:
                assert_equal(t_obtained_exog_coint, t_obtained, 'WITH EXOG_COINT' + err_msg)
            p_obtained = results_sm[ds][dt].pvalues_det_coef_coint
            p_obtained_exog = results_sm_exog[ds][dt].pvalues_det_coef_coint
            p_obtained_exog_coint = results_sm_exog_coint[ds][dt].pvalues_det_coef_coint
            p_desired = results_ref[ds][dt]['p']['det_coint']
            assert_allclose(p_obtained, p_desired, rtol, atol, False, 'p-VALUES\n' + err_msg)
            if exog:
                assert_equal(p_obtained_exog, p_obtained, 'WITH EXOG' + err_msg)
            if exog_coint:
                assert_equal(p_obtained_exog_coint, p_obtained, 'WITH EXOG_COINT' + err_msg)