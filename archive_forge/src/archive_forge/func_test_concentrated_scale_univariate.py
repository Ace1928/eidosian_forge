import numpy as np
import pandas as pd
from statsmodels.tools.tools import Bunch
from .results import results_varmax
from statsmodels.tsa.statespace import sarimax, varmax
from numpy.testing import assert_raises, assert_allclose
def test_concentrated_scale_univariate():
    check_concentrated_scale(filter_univariate=True)
    check_concentrated_scale(filter_univariate=True, measurement_error=True)
    check_concentrated_scale(filter_univariate=True, error_cov_type='diagonal')
    check_concentrated_scale(filter_univariate=True, missing=True)
    check_concentrated_scale(filter_univariate=True, missing=True, loglikelihood_burn=10)