import warnings
import numpy as np
from numpy.testing import assert_equal, assert_allclose, assert_raises
import pandas as pd
import pytest
from statsmodels.datasets import macrodata
from statsmodels.tools.sm_exceptions import SpecificationWarning
from statsmodels.tsa.statespace import structural
from statsmodels.tsa.statespace.structural import UnobservedComponents
from statsmodels.tsa.statespace.tests.results import results_structural
@pytest.mark.slow
def test_lltrend_cycle_seasonal_reg_ar1(close_figures):
    run_ucm('lltrend_cycle_seasonal_reg_ar1_approx_diffuse')
    run_ucm('lltrend_cycle_seasonal_reg_ar1', use_exact_diffuse=True)