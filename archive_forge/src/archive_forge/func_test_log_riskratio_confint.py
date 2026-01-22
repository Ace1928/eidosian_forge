import os
import warnings
import numpy as np
import statsmodels.stats.contingency_tables as ctab
import pandas as pd
from numpy.testing import assert_allclose, assert_equal
import statsmodels.api as sm
def test_log_riskratio_confint(self):
    lcb1, ucb1 = self.tbl_obj.log_riskratio_confint(0.05)
    lcb2, ucb2 = self.log_riskratio_confint
    assert_allclose(lcb1, lcb2)
    assert_allclose(ucb1, ucb2)