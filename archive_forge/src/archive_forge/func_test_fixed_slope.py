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
def test_fixed_slope(close_figures):
    warning = SpecificationWarning
    match = 'irregular component added'
    with pytest.warns(warning, match=match):
        run_ucm('fixed_slope')
        run_ucm('fixed_slope', use_exact_diffuse=True)