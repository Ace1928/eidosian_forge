import os
import warnings
import numpy as np
import statsmodels.stats.contingency_tables as ctab
import pandas as pd
from numpy.testing import assert_allclose, assert_equal
import statsmodels.api as sm
def test_chi2_association():
    np.random.seed(8743)
    table = np.random.randint(10, 30, size=(4, 4))
    from scipy.stats import chi2_contingency
    rslt_scipy = chi2_contingency(table)
    b = ctab.Table(table).test_nominal_association()
    assert_allclose(b.statistic, rslt_scipy[0])
    assert_allclose(b.pvalue, rslt_scipy[1])