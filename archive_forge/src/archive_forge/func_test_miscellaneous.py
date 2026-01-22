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
def test_miscellaneous():
    exog = np.arange(75)
    mod = CheckDynamicFactor()
    mod.setup_class(true=None, k_factors=1, factor_order=1, exog=exog, filter=False)
    exog = pd.Series(np.arange(75), index=pd.date_range(start='1960-04-01', end='1978-10-01', freq='QS'))
    mod = CheckDynamicFactor()
    mod.setup_class(true=None, k_factors=1, factor_order=1, exog=exog, filter=False)