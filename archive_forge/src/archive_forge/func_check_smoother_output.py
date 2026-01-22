import numpy as np
import pytest
from statsmodels.tsa.statespace import (
from statsmodels.tsa.statespace.tests.test_impulse_responses import TVSS
from numpy.testing import assert_allclose
def check_smoother_output(mod, periods, atol=1e-12):
    if isinstance(mod, mlemodel.MLEModel):
        res_mv = mod.ssm.smooth()
        kfilter = mod.ssm._kalman_filter
        kfilter.seek(0, True)
        kfilter.univariate_filter[periods] = 1
        for _ in range(mod.nobs):
            next(kfilter)
        res_switch = mod.ssm.results_class(mod.ssm)
        res_switch.update_representation(mod.ssm)
        res_switch.update_filter(kfilter)
        mod.ssm._kalman_smoother.reset(True)
        smoother = mod.ssm._smooth()
        res_switch.update_smoother(smoother)
        mod.ssm.filter_univariate = True
        res_uv = mod.ssm.smooth()
    else:
        res_mv, res_switch, res_uv = mod
    attrs = ['scaled_smoothed_estimator', 'scaled_smoothed_estimator_cov', 'smoothed_state', 'smoothed_state_cov', 'smoothed_state_autocov', 'smoothed_state_disturbance', 'smoothed_state_disturbance_cov', 'innovations_transition']
    for attr in attrs:
        attr_mv = getattr(res_mv, attr)
        attr_uv = getattr(res_uv, attr)
        attr_switch = getattr(res_switch, attr)
        if attr_mv is None:
            continue
        assert_allclose(attr_uv, attr_mv, atol=atol)
        assert_allclose(attr_switch, attr_mv, atol=atol)
        assert_allclose(attr_switch, attr_uv, atol=atol)
    attrs = ['smoothing_error', 'smoothed_measurement_disturbance', 'smoothed_measurement_disturbance_cov']
    for attr in attrs:
        attr_mv = getattr(res_mv, attr)
        attr_uv = getattr(res_uv, attr)
        attr_switch = getattr(res_switch, attr)
        if attr_mv is None:
            continue
        actual = attr_switch.copy()
        desired = attr_mv.copy()
        actual[..., periods] = 0
        desired[..., periods] = 0
        assert_allclose(actual, desired)