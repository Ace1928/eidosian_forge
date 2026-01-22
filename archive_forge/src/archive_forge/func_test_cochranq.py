import os
import warnings
import numpy as np
import statsmodels.stats.contingency_tables as ctab
import pandas as pd
from numpy.testing import assert_allclose, assert_equal
import statsmodels.api as sm
def test_cochranq():
    table = [[1, 0, 1, 1], [0, 1, 1, 1], [1, 1, 1, 0], [0, 1, 0, 0], [0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 0], [1, 1, 1, 1], [0, 1, 0, 0]]
    table = np.asarray(table)
    stat, pvalue, df = ctab.cochrans_q(table, return_object=False)
    assert_allclose(stat, 4.2)
    assert_allclose(df, 3)
    table = [[0, 0, 1, 1, 0], [0, 1, 0, 1, 0], [0, 1, 1, 0, 1], [1, 0, 0, 0, 1], [1, 1, 0, 0, 0], [1, 0, 1, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 1, 0], [0, 0, 0, 0, 0]]
    table = np.asarray(table)
    stat, pvalue, df = ctab.cochrans_q(table, return_object=False)
    assert_allclose(stat, 1.2174, rtol=0.0001)
    assert_allclose(df, 4)
    data = table[:, 0:2]
    xtab = np.asarray(pd.crosstab(data[:, 0], data[:, 1]))
    b1 = ctab.cochrans_q(data, return_object=True)
    b2 = ctab.mcnemar(xtab, exact=False, correction=False)
    assert_allclose(b1.statistic, b2.statistic)
    assert_allclose(b1.pvalue, b2.pvalue)
    assert_equal(str(b1).startswith('df          1\npvalue      0.65'), True)