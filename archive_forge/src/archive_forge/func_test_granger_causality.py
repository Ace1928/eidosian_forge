import numpy as np
from numpy.testing import (
import pytest
import statsmodels.datasets.interest_inflation.data as e6
from statsmodels.tools.testing import assert_equal
from statsmodels.tsa.vector_ar.tests.JMulTi_results.parse_jmulti_vecm_output import (
from statsmodels.tsa.vector_ar.util import seasonal_dummies
from statsmodels.tsa.vector_ar.var_model import VARProcess
from statsmodels.tsa.vector_ar.vecm import (
def test_granger_causality():
    if debug_mode:
        if 'granger' not in to_test:
            return
        else:
            print('\n\nGRANGER', end='')
    for ds in datasets:
        for dt in ds.dt_s_list:
            if debug_mode:
                print('\n' + dt_s_tup_to_string(dt) + ': ', end='')
            exog = results_sm_exog[ds][dt].exog is not None
            exog_coint = results_sm_exog_coint[ds][dt].exog_coint is not None
            err_msg_g_p = build_err_msg(ds, dt, 'GRANGER CAUS. - p-VALUE')
            err_msg_g_t = build_err_msg(ds, dt, 'GRANGER CAUS. - TEST STAT.')
            v_ind = range(len(ds.variable_names))
            for causing_ind in sublists(v_ind, 1, len(v_ind) - 1):
                causing_names = ['y' + str(i + 1) for i in causing_ind]
                causing_key = tuple((ds.variable_names[i] for i in causing_ind))
                caused_ind = [i for i in v_ind if i not in causing_ind]
                caused_names = ['y' + str(i + 1) for i in caused_ind]
                caused_key = tuple((ds.variable_names[i] for i in caused_ind))
                granger_sm_ind = results_sm[ds][dt].test_granger_causality(caused_ind, causing_ind)
                granger_sm_ind_exog = results_sm_exog[ds][dt].test_granger_causality(caused_ind, causing_ind)
                granger_sm_ind_exog_coint = results_sm_exog_coint[ds][dt].test_granger_causality(caused_ind, causing_ind)
                granger_sm_str = results_sm[ds][dt].test_granger_causality(caused_names, causing_names)
                granger_sm_ind.summary()
                str(granger_sm_ind)
                assert_(granger_sm_ind == granger_sm_str)
                g_t_obt = granger_sm_ind.test_statistic
                g_t_obt_exog = granger_sm_ind_exog.test_statistic
                g_t_obt_exog_coint = granger_sm_ind_exog_coint.test_statistic
                g_t_des = results_ref[ds][dt]['granger_caus']['test_stat'][causing_key, caused_key]
                assert_allclose(g_t_obt, g_t_des, rtol, atol, False, err_msg_g_t)
                if exog:
                    assert_allclose(g_t_obt_exog, g_t_obt, 1e-07, 0, False, 'WITH EXOG' + err_msg_g_t)
                if exog_coint:
                    assert_allclose(g_t_obt_exog_coint, g_t_obt, 1e-07, 0, False, 'WITH EXOG_COINT' + err_msg_g_t)
                g_t_obt_str = granger_sm_str.test_statistic
                assert_allclose(g_t_obt_str, g_t_obt, 1e-07, 0, False, err_msg_g_t + ' - sequences of integers and '.upper() + 'strings as arguments do not yield the same result!'.upper())
                if len(causing_ind) == 1 or len(caused_ind) == 1:
                    ci = causing_ind[0] if len(causing_ind) == 1 else causing_ind
                    ce = caused_ind[0] if len(caused_ind) == 1 else caused_ind
                    granger_sm_single_ind = results_sm[ds][dt].test_granger_causality(ce, ci)
                    g_t_obt_single = granger_sm_single_ind.test_statistic
                    assert_allclose(g_t_obt_single, g_t_obt, 1e-07, 0, False, err_msg_g_t + ' - list of int and int as '.upper() + 'argument do not yield the same result!'.upper())
                g_p_obt = granger_sm_ind.pvalue
                g_p_des = results_ref[ds][dt]['granger_caus']['p'][causing_key, caused_key]
                assert_allclose(g_p_obt, g_p_des, rtol, atol, False, err_msg_g_p)
                g_p_obt_str = granger_sm_str.pvalue
                assert_allclose(g_p_obt_str, g_p_obt, 1e-07, 0, False, err_msg_g_t + ' - sequences of integers and '.upper() + 'strings as arguments do not yield the same result!'.upper())
                if len(causing_ind) == 1:
                    g_p_obt_single = granger_sm_single_ind.pvalue
                    assert_allclose(g_p_obt_single, g_p_obt, 1e-07, 0, False, err_msg_g_t + ' - list of int and int as '.upper() + 'argument do not yield the same result!'.upper())