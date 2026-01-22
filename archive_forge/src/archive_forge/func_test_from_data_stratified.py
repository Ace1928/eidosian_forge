import os
import warnings
import numpy as np
import statsmodels.stats.contingency_tables as ctab
import pandas as pd
from numpy.testing import assert_allclose, assert_equal
import statsmodels.api as sm
def test_from_data_stratified():
    df = pd.DataFrame([[1, 1, 1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1, 0, 0], [0, 0, 0, 0, 1, 1, 1, 1]]).T
    e = np.asarray([[[0, 1], [1, 1]], [[2, 2], [1, 0]]])
    tab1 = ctab.StratifiedTable.from_data(0, 1, 2, df)
    assert_equal(tab1.table, e)
    tab1 = ctab.StratifiedTable.from_data(0, 1, 2, np.asarray(df))
    assert_equal(tab1.table, e)