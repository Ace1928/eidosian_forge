from statsmodels.compat.pandas import MONTH_END
import os
import pickle
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from statsmodels.datasets import co2
from statsmodels.tsa.seasonal import STL, DecomposeResult
def test_parameter_checks_period(default_kwargs):
    class_kwargs, _, _ = _to_class_kwargs(default_kwargs)
    endog = class_kwargs['endog']
    endog2 = np.hstack((endog[:, None], endog[:, None]))
    period = class_kwargs['period']
    with pytest.raises(ValueError, match='endog is required to have ndim 1'):
        STL(endog=endog2, period=period)
    match = 'period must be a positive integer >= 2'
    with pytest.raises(ValueError, match=match):
        STL(endog=endog, period=1)
    with pytest.raises(ValueError, match=match):
        STL(endog=endog, period=-12)
    with pytest.raises(ValueError, match=match):
        STL(endog=endog, period=4.0)