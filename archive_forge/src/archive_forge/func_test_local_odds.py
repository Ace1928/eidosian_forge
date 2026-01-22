import os
import warnings
import numpy as np
import statsmodels.stats.contingency_tables as ctab
import pandas as pd
from numpy.testing import assert_allclose, assert_equal
import statsmodels.api as sm
def test_local_odds():
    table = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    table = np.asarray(table)
    tbl_obj = ctab.Table(table)
    loc_odds = tbl_obj.local_oddsratios
    assert_allclose(loc_odds[0, 0], 5 / 8.0)
    assert_allclose(loc_odds[0, 1], 12 / float(15), atol=1e-05, rtol=1e-05)
    assert_allclose(np.log(loc_odds), tbl_obj.local_log_oddsratios, atol=1e-05, rtol=1e-05)