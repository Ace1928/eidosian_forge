import numpy as np
from numpy.testing import (
import pytest
import statsmodels.datasets.interest_inflation.data as e6
from statsmodels.tools.testing import assert_equal
from statsmodels.tsa.vector_ar.tests.JMulTi_results.parse_jmulti_vecm_output import (
from statsmodels.tsa.vector_ar.util import seasonal_dummies
from statsmodels.tsa.vector_ar.var_model import VARProcess
from statsmodels.tsa.vector_ar.vecm import (
def test_inst_causality():
    if debug_mode:
        if 'inst. causality' not in to_test:
            return
        else:
            print('\n\nINST. CAUSALITY', end='')
    for ds in datasets:
        for dt in ds.dt_s_list:
            if debug_mode:
                print('\n' + dt_s_tup_to_string(dt) + ': ', end='')
            exog = results_sm_exog[ds][dt].exog is not None
            exog_coint = results_sm_exog_coint[ds][dt].exog_coint is not None
            err_msg_i_p = build_err_msg(ds, dt, 'INSTANT. CAUS. - p-VALUE')
            err_msg_i_t = build_err_msg(ds, dt, 'INSTANT. CAUS. - TEST STAT.')
            v_ind = range(len(ds.variable_names))
            for causing_ind in sublists(v_ind, 1, len(v_ind) - 1):
                causing_names = ['y' + str(i + 1) for i in causing_ind]
                causing_key = tuple((ds.variable_names[i] for i in causing_ind))
                caused_ind = [i for i in v_ind if i not in causing_ind]
                caused_key = tuple((ds.variable_names[i] for i in caused_ind))
                inst_sm_ind = results_sm[ds][dt].test_inst_causality(causing_ind)
                inst_sm_ind_exog = results_sm_exog[ds][dt].test_inst_causality(causing_ind)
                inst_sm_ind_exog_coint = results_sm_exog_coint[ds][dt].test_inst_causality(causing_ind)
                inst_sm_str = results_sm[ds][dt].test_inst_causality(causing_names)
                inst_sm_ind.summary()
                str(inst_sm_ind)
                assert_(inst_sm_ind == inst_sm_str)
                t_obt = inst_sm_ind.test_statistic
                t_obt_exog = inst_sm_ind_exog.test_statistic
                t_obt_exog_coint = inst_sm_ind_exog_coint.test_statistic
                t_des = results_ref[ds][dt]['inst_caus']['test_stat'][causing_key, caused_key]
                assert_allclose(t_obt, t_des, rtol, atol, False, err_msg_i_t)
                if exog:
                    assert_allclose(t_obt_exog, t_obt, 1e-07, 0, False, 'WITH EXOG' + err_msg_i_t)
                if exog_coint:
                    assert_allclose(t_obt_exog_coint, t_obt, 1e-07, 0, False, 'WITH EXOG_COINT' + err_msg_i_t)
                t_obt_str = inst_sm_str.test_statistic
                assert_allclose(t_obt_str, t_obt, 1e-07, 0, False, err_msg_i_t + ' - sequences of integers and '.upper() + 'strings as arguments do not yield the same result!'.upper())
                if len(causing_ind) == 1:
                    inst_sm_single_ind = results_sm[ds][dt].test_inst_causality(causing_ind[0])
                    t_obt_single = inst_sm_single_ind.test_statistic
                    assert_allclose(t_obt_single, t_obt, 1e-07, 0, False, err_msg_i_t + ' - list of int and int as '.upper() + 'argument do not yield the same result!'.upper())
                p_obt = results_sm[ds][dt].test_inst_causality(causing_ind).pvalue
                p_des = results_ref[ds][dt]['inst_caus']['p'][causing_key, caused_key]
                assert_allclose(p_obt, p_des, rtol, atol, False, err_msg_i_p)
                p_obt_str = inst_sm_str.pvalue
                assert_allclose(p_obt_str, p_obt, 1e-07, 0, False, err_msg_i_p + ' - sequences of integers and '.upper() + 'strings as arguments do not yield the same result!'.upper())
                if len(causing_ind) == 1:
                    inst_sm_single_ind = results_sm[ds][dt].test_inst_causality(causing_ind[0])
                    p_obt_single = inst_sm_single_ind.pvalue
                    assert_allclose(p_obt_single, p_obt, 1e-07, 0, False, err_msg_i_p + ' - list of int and int as '.upper() + 'argument do not yield the same result!'.upper())