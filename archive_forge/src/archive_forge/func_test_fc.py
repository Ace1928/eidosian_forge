import numpy as np
from numpy.testing import (
import pytest
import statsmodels.datasets.interest_inflation.data as e6
from statsmodels.tools.testing import assert_equal
from statsmodels.tsa.vector_ar.tests.JMulTi_results.parse_jmulti_vecm_output import (
from statsmodels.tsa.vector_ar.util import seasonal_dummies
from statsmodels.tsa.vector_ar.var_model import VARProcess
from statsmodels.tsa.vector_ar.vecm import (
def test_fc():
    if debug_mode:
        if 'fc' not in to_test:
            return
        else:
            print('\n\nFORECAST', end='')
    for ds in datasets:
        for dt in ds.dt_s_list:
            if debug_mode:
                print('\n' + dt_s_tup_to_string(dt) + ': ', end='')
            STEPS = 5
            ALPHA = 0.05
            err_msg = build_err_msg(ds, dt, 'FORECAST')
            obtained = results_sm[ds][dt].predict(steps=STEPS)
            desired = results_ref[ds][dt]['fc']['fc']
            assert_allclose(obtained, desired, rtol, atol, False, err_msg)
            exog = results_sm_exog[ds][dt].exog is not None
            exog_fc = None
            if exog:
                seasons = dt[1]
                exog_model = results_sm_exog[ds][dt].exog
                exog_seasons_fc = exog_model[-seasons:, :seasons - 1]
                exog_seasons_fc = np.pad(exog_seasons_fc, ((0, STEPS - exog_seasons_fc.shape[0]), (0, 0)), 'wrap')
                if exog_seasons_fc.shape[1] + 1 == exog_model.shape[1]:
                    exog_lt_fc = exog_model[-1, -1] + 1 + np.arange(STEPS)
                    exog_fc = np.column_stack((exog_seasons_fc, exog_lt_fc))
                else:
                    exog_fc = exog_seasons_fc
                obtained_exog = results_sm_exog[ds][dt].predict(steps=STEPS, exog_fc=exog_fc)
                assert_allclose(obtained_exog, obtained, 1e-07, 0, False, 'WITH EXOG' + err_msg)
            err_msg = build_err_msg(ds, dt, 'FORECAST WITH INTERVALS')
            obtained_w_intervals = results_sm[ds][dt].predict(steps=STEPS, alpha=ALPHA)
            obtained_w_intervals_exog = results_sm_exog[ds][dt].predict(steps=STEPS, alpha=ALPHA, exog_fc=exog_fc)
            obt = obtained_w_intervals[0]
            obt_l = obtained_w_intervals[1]
            obt_u = obtained_w_intervals[2]
            obt_exog = obtained_w_intervals_exog[0]
            obt_exog_l = obtained_w_intervals_exog[1]
            obt_exog_u = obtained_w_intervals_exog[2]
            des = results_ref[ds][dt]['fc']['fc']
            des_l = results_ref[ds][dt]['fc']['lower']
            des_u = results_ref[ds][dt]['fc']['upper']
            assert_allclose(obt, des, rtol, atol, False, err_msg)
            assert_allclose(obt_l, des_l, rtol, atol, False, err_msg)
            assert_allclose(obt_u, des_u, rtol, atol, False, err_msg)
            if exog:
                assert_allclose(obt_exog, obt, 1e-07, 0, False, 'WITH EXOG' + err_msg)
                assert_allclose(obt_exog_l, obt_l, 1e-07, 0, False, 'WITH EXOG' + err_msg)
                assert_allclose(obt_exog_u, obt_u, 1e-07, 0, False, 'WITH EXOG' + err_msg)
            exog_coint_model = results_sm_exog_coint[ds][dt].exog_coint
            exog_coint = exog_coint_model is not None
            exog_coint_fc = None
            if exog_coint:
                exog_coint_fc = np.ones(STEPS - 1)
                if exog_coint_model.shape[1] == 2:
                    exog_coint_fc = np.column_stack((exog_coint_fc, exog_coint_model[-1, -1] + 1 + np.arange(STEPS - 1)))
                obtained_exog_coint = results_sm_exog_coint[ds][dt].predict(steps=STEPS, exog_coint_fc=exog_coint_fc)
                assert_allclose(obtained_exog_coint, obtained, 1e-07, 0, False, 'WITH EXOG_COINT' + err_msg)
            err_msg = build_err_msg(ds, dt, 'FORECAST WITH INTERVALS')
            obtained_w_intervals = results_sm[ds][dt].predict(steps=STEPS, alpha=ALPHA)
            obtained_w_intervals_exog_coint = results_sm_exog_coint[ds][dt].predict(steps=STEPS, alpha=ALPHA, exog_coint_fc=exog_coint_fc)
            obt = obtained_w_intervals[0]
            obt_l = obtained_w_intervals[1]
            obt_u = obtained_w_intervals[2]
            obt_exog_coint = obtained_w_intervals_exog_coint[0]
            obt_exog_coint_l = obtained_w_intervals_exog_coint[1]
            obt_exog_coint_u = obtained_w_intervals_exog_coint[2]
            des = results_ref[ds][dt]['fc']['fc']
            des_l = results_ref[ds][dt]['fc']['lower']
            des_u = results_ref[ds][dt]['fc']['upper']
            assert_allclose(obt, des, rtol, atol, False, err_msg)
            assert_allclose(obt_l, des_l, rtol, atol, False, err_msg)
            assert_allclose(obt_u, des_u, rtol, atol, False, err_msg)
            if exog_coint:
                assert_allclose(obt_exog_coint, obt, 1e-07, 0, False, 'WITH EXOG_COINT' + err_msg)
                assert_allclose(obt_exog_coint_l, obt_l, 1e-07, 0, False, 'WITH EXOG_COINT' + err_msg)
                assert_allclose(obt_exog_coint_u, obt_u, 1e-07, 0, False, 'WITH EXOG_COINT' + err_msg)