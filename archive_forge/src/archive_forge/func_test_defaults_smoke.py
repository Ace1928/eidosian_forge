from statsmodels.compat.pandas import MONTH_END
import os
import pickle
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from statsmodels.datasets import co2
from statsmodels.tsa.seasonal import STL, DecomposeResult
def test_defaults_smoke(default_kwargs, robust):
    class_kwargs, _, _ = _to_class_kwargs(default_kwargs, robust)
    endog = class_kwargs['endog']
    period = class_kwargs['period']
    mod = STL(endog=endog, period=period)
    mod.fit()