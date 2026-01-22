import os
import re
import warnings
import numpy as np
from numpy.testing import assert_equal, assert_raises, assert_allclose
import pandas as pd
import pytest
from statsmodels.tsa.statespace import dynamic_factor
from .results import results_varmax, results_dynamic_factor
from statsmodels.iolib.summary import forg
def test_recreate_model():
    nobs = 100
    endog = np.ones((nobs, 3)) * 2.0
    exog = np.ones(nobs)
    k_factors = [0, 1, 2]
    factor_orders = [0, 1, 2]
    error_orders = [0, 1]
    error_vars = [False, True]
    error_cov_types = ['diagonal', 'scalar']
    import itertools
    names = ['k_factors', 'factor_order', 'error_order', 'error_var', 'error_cov_type']
    for element in itertools.product(k_factors, factor_orders, error_orders, error_vars, error_cov_types):
        kwargs = dict(zip(names, element))
        mod = dynamic_factor.DynamicFactor(endog, exog=exog, **kwargs)
        mod2 = dynamic_factor.DynamicFactor(endog, exog=exog, **mod._get_init_kwds())
        check_equivalent_models(mod, mod2)