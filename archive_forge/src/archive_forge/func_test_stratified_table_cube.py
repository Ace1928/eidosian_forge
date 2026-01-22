import os
import warnings
import numpy as np
import statsmodels.stats.contingency_tables as ctab
import pandas as pd
from numpy.testing import assert_allclose, assert_equal
import statsmodels.api as sm
def test_stratified_table_cube():
    tab1 = [[[8, 9], [6, 7]], [[4, 9], [5, 5]], [[8, 8], [9, 11]]]
    tab2 = np.asarray(tab1).T
    ct1 = ctab.StratifiedTable(tab1)
    ct2 = ctab.StratifiedTable(tab2)
    assert_allclose(ct1.oddsratio_pooled, ct2.oddsratio_pooled)
    assert_allclose(ct1.logodds_pooled, ct2.logodds_pooled)