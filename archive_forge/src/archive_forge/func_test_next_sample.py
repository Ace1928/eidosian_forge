import numpy as np
import pandas as pd
import pytest
from statsmodels.imputation import mice
import statsmodels.api as sm
from numpy.testing import assert_equal, assert_allclose
import warnings
def test_next_sample(self):
    df = gendat()
    imp_data = mice.MICEData(df)
    all_x = []
    for j in range(2):
        x = imp_data.next_sample()
        assert isinstance(x, pd.DataFrame)
        assert_equal(df.shape, x.shape)
        all_x.append(x)
    assert all_x[0] is all_x[1]