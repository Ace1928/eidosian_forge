import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_equal, assert_allclose, assert_raises
from statsmodels.tsa.statespace.tools import (
from statsmodels.tsa.arima import specification
@pytest.mark.parametrize('n,d,D,s,params,which', [(0, 0, 0, 0, np.array([1.0]), 'p'), (1, 0, 0, 0, np.array([0.5, 1.0]), 'p'), (1, 0, 0, 0, np.array([-0.2, 100.0]), 'p'), (2, 0, 0, 0, np.array([-0.2, 0.5, 100.0]), 'p'), (20, 0, 0, 0, np.array([0.0] * 20 + [100.0]), 'p'), (0, 1, 0, 0, np.array([1.0]), 'p'), (0, 1, 1, 4, np.array([1.0]), 'p'), (1, 1, 0, 0, np.array([0.5, 1.0]), 'p'), (1, 1, 1, 4, np.array([0.5, 1.0]), 'p'), (0, 0, 0, 0, np.array([1.0]), 'q'), (1, 0, 0, 0, np.array([0.5, 1.0]), 'q'), (1, 0, 0, 0, np.array([-0.2, 100.0]), 'q'), (2, 0, 0, 0, np.array([-0.2, 0.5, 100.0]), 'q'), (20, 0, 0, 0, np.array([0.0] * 20 + [100.0]), 'q'), (0, 1, 0, 0, np.array([1.0]), 'q'), (0, 1, 1, 4, np.array([1.0]), 'q'), (1, 1, 0, 0, np.array([0.5, 1.0]), 'q'), (1, 1, 1, 4, np.array([0.5, 1.0]), 'q')])
def test_specification_ar_or_ma(n, d, D, s, params, which):
    if which == 'p':
        p, d, q = (n, d, 0)
        ar_names = ['ar.L%d' % i for i in range(1, p + 1)]
        ma_names = []
    else:
        p, d, q = (0, d, n)
        ar_names = []
        ma_names = ['ma.L%d' % i for i in range(1, q + 1)]
    ar_params = params[:p]
    ma_params = params[p:-1]
    sigma2 = params[-1]
    P, D, Q, s = (0, D, 0, s)
    args = ((p, d, q), (P, D, Q, s))
    kwargs = {'enforce_stationarity': None, 'enforce_invertibility': None, 'concentrate_scale': None}
    properties_kwargs = kwargs.copy()
    properties_kwargs.update({'is_ar_consecutive': True, 'is_ma_consecutive': True, 'exog_names': [], 'ar_names': ar_names, 'ma_names': ma_names, 'seasonal_ar_names': [], 'seasonal_ma_names': []})
    methods_kwargs = kwargs.copy()
    methods_kwargs.update({'exog_params': [], 'ar_params': ar_params, 'ma_params': ma_params, 'seasonal_ar_params': [], 'seasonal_ma_params': [], 'sigma2': sigma2})
    spec = specification.SARIMAXSpecification(order=(p, d, q), seasonal_order=(P, D, Q, s))
    check_attributes(spec, *args, **kwargs)
    check_properties(spec, *args, **properties_kwargs)
    check_methods(spec, *args, **methods_kwargs)
    spec = specification.SARIMAXSpecification(ar_order=p, diff=d, ma_order=q, seasonal_ar_order=P, seasonal_diff=D, seasonal_ma_order=Q, seasonal_periods=s)
    check_attributes(spec, *args, **kwargs)
    check_properties(spec, *args, **properties_kwargs)
    check_methods(spec, *args, **methods_kwargs)