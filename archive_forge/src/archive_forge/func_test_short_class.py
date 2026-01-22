from statsmodels.compat.pandas import MONTH_END
import os
import pickle
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from statsmodels.datasets import co2
from statsmodels.tsa.seasonal import STL, DecomposeResult
def test_short_class(default_kwargs_short):
    class_kwargs, outer, inner = _to_class_kwargs(default_kwargs_short)
    mod = STL(**class_kwargs)
    res = mod.fit(outer_iter=outer, inner_iter=inner)
    expected = results.loc['short'].sort_index()
    assert_allclose(res.seasonal, expected.season)
    assert_allclose(res.trend, expected.trend)
    assert_allclose(res.weights, expected.rw)