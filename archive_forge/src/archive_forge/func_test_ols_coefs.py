import numpy as np
from numpy.testing import assert_, assert_allclose, assert_raises
import statsmodels.datasets.macrodata.data as macro
from statsmodels.tsa.vector_ar.tests.JMulTi_results.parse_jmulti_vecm_output import (
from statsmodels.tsa.vector_ar.var_model import VAR
from .JMulTi_results.parse_jmulti_var_output import (
def test_ols_coefs():
    if debug_mode:
        if 'coefs' not in to_test:
            return
        print('\n\nESTIMATED PARAMETER MATRICES FOR LAGGED ENDOG', end='')
    for ds in datasets:
        for dt_s in dt_s_list:
            if debug_mode:
                print('\n' + dt_s_tup_to_string(dt_s) + ': ', end='')
            err_msg = build_err_msg(ds, dt_s, 'PARAMETER MATRICES ENDOG')
            obtained = np.hstack(results_sm[ds][dt_s].coefs)
            desired = results_ref[ds][dt_s]['est']['Lagged endogenous term']
            assert_allclose(obtained, desired, rtol, atol, False, err_msg)
            if debug_mode and dont_test_se_t_p:
                continue
            obt = results_sm[ds][dt_s].stderr_endog_lagged
            des = results_ref[ds][dt_s]['se']['Lagged endogenous term'].T
            assert_allclose(obt, des, rtol, atol, False, 'STANDARD ERRORS\n' + err_msg)
            obt = results_sm[ds][dt_s].tvalues_endog_lagged
            des = results_ref[ds][dt_s]['t']['Lagged endogenous term'].T
            assert_allclose(obt, des, rtol, atol, False, 't-VALUES\n' + err_msg)
            obt = results_sm[ds][dt_s].pvalues_endog_lagged
            des = results_ref[ds][dt_s]['p']['Lagged endogenous term'].T
            assert_allclose(obt, des, rtol, atol, False, 'p-VALUES\n' + err_msg)