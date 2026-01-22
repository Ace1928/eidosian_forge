import os
import warnings
import numpy as np
import statsmodels.stats.contingency_tables as ctab
import pandas as pd
from numpy.testing import assert_allclose, assert_equal
import statsmodels.api as sm
def test_from_data_2x2():
    df = pd.DataFrame([[1, 1, 1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1, 0, 0]]).T
    e = np.asarray([[1, 2], [4, 1]])
    tab1 = ctab.Table2x2.from_data(df, shift_zeros=False)
    assert_equal(tab1.table, e)
    tab1 = ctab.Table2x2.from_data(np.asarray(df), shift_zeros=False)
    assert_equal(tab1.table, e)