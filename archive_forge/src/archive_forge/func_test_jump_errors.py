from statsmodels.compat.pandas import MONTH_END
import os
import pickle
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from statsmodels.datasets import co2
from statsmodels.tsa.seasonal import STL, DecomposeResult
def test_jump_errors(default_kwargs):
    class_kwargs, _, _ = _to_class_kwargs(default_kwargs)
    endog = class_kwargs['endog']
    period = class_kwargs['period']
    with pytest.raises(ValueError, match='low_pass_jump must be a positive'):
        STL(endog=endog, period=period, low_pass_jump=0)
    with pytest.raises(ValueError, match='low_pass_jump must be a positive'):
        STL(endog=endog, period=period, low_pass_jump=1.0)
    with pytest.raises(ValueError, match='seasonal_jump must be a positive'):
        STL(endog=endog, period=period, seasonal_jump=0)
    with pytest.raises(ValueError, match='seasonal_jump must be a positive'):
        STL(endog=endog, period=period, seasonal_jump=1.0)
    with pytest.raises(ValueError, match='trend_jump must be a positive'):
        STL(endog=endog, period=period, trend_jump=0)
    with pytest.raises(ValueError, match='trend_jump must be a positive'):
        STL(endog=endog, period=period, trend_jump=1.0)