import os
import warnings
import numpy as np
import statsmodels.stats.contingency_tables as ctab
import pandas as pd
from numpy.testing import assert_allclose, assert_equal
import statsmodels.api as sm
def test_resids():
    table = [[12, 8, 31, 41], [307, 246, 439, 245]]
    fit = [[22.083, 17.583, 32.536, 19.798], [296.92, 236.42, 437.46, 266.2]]
    c2 = [[4.6037, 5.223, 0.0725, 22.704], [0.3424, 0.3885, 0.0054, 1.6886]]
    pr = np.array([[-2.14562121, -2.28538719, -0.26923882, 4.7649169], [0.58514314, 0.62325942, 0.07342547, -1.29946443]])
    sr = np.array([[-2.55112945, -2.6338782, -0.34712127, 5.5751083], [2.55112945, 2.6338782, 0.34712127, -5.5751083]])
    tab = ctab.Table(table)
    assert_allclose(tab.fittedvalues, fit, atol=0.0001, rtol=0.0001)
    assert_allclose(tab.chi2_contribs, c2, atol=0.0001, rtol=0.0001)
    assert_allclose(tab.resid_pearson, pr, atol=0.0001, rtol=0.0001)
    assert_allclose(tab.standardized_resids, sr, atol=0.0001, rtol=0.0001)