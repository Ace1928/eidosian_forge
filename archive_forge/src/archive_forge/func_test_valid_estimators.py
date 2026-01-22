import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_equal, assert_allclose, assert_raises
from statsmodels.tsa.statespace.tools import (
from statsmodels.tsa.arima import specification
@pytest.mark.parametrize('order,seasonal_order,enforce_stationarity,enforce_invertibility,concentrate_scale,valid', [((0, 0, 0), (0, 0, 0, 0), None, None, None, ['yule_walker', 'burg', 'innovations', 'hannan_rissanen', 'innovations_mle', 'statespace']), ((1, 0, 0), (0, 0, 0, 0), None, None, None, ['yule_walker', 'burg', 'hannan_rissanen', 'innovations_mle', 'statespace']), ((0, 0, 1), (0, 0, 0, 0), None, None, None, ['innovations', 'hannan_rissanen', 'innovations_mle', 'statespace']), ((1, 0, 1), (0, 0, 0, 0), None, None, None, ['hannan_rissanen', 'innovations_mle', 'statespace']), ((0, 0, 0), (1, 0, 0, 4), None, None, None, ['innovations_mle', 'statespace']), ((1, 0, 0), (0, 0, 0, 0), True, None, None, ['innovations_mle', 'statespace']), ((1, 0, 0), (0, 0, 0, 0), False, None, None, ['yule_walker', 'burg', 'hannan_rissanen', 'statespace']), ((1, 0, 0), (0, 0, 0, 0), None, True, None, ['yule_walker', 'burg', 'hannan_rissanen', 'innovations_mle', 'statespace']), ((1, 0, 0), (0, 0, 0, 0), None, False, None, ['yule_walker', 'burg', 'hannan_rissanen', 'innovations_mle', 'statespace']), ((1, 0, 0), (0, 0, 0, 0), None, None, True, ['yule_walker', 'burg', 'hannan_rissanen', 'statespace'])])
def test_valid_estimators(order, seasonal_order, enforce_stationarity, enforce_invertibility, concentrate_scale, valid):
    spec = specification.SARIMAXSpecification(order=order, seasonal_order=seasonal_order, enforce_stationarity=enforce_stationarity, enforce_invertibility=enforce_invertibility, concentrate_scale=concentrate_scale)
    estimators = {'yule_walker', 'burg', 'innovations', 'hannan_rissanen', 'innovations_mle', 'statespace'}
    desired = set(valid)
    assert_equal(spec.valid_estimators, desired)
    for estimator in desired:
        assert_equal(spec.validate_estimator(estimator), None)
    for estimator in estimators.difference(desired):
        print(estimator, enforce_stationarity)
        assert_raises(ValueError, spec.validate_estimator, estimator)
    spec = specification.SARIMAXSpecification(endog=[np.nan], order=order, seasonal_order=seasonal_order, enforce_stationarity=enforce_stationarity, enforce_invertibility=enforce_invertibility, concentrate_scale=concentrate_scale)
    assert_equal(spec.valid_estimators, {'statespace'})
    assert_equal(spec.validate_estimator('statespace'), None)
    for estimator in estimators.difference(['statespace']):
        assert_raises(ValueError, spec.validate_estimator, estimator)