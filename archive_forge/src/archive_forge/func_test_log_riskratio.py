import os
import warnings
import numpy as np
import statsmodels.stats.contingency_tables as ctab
import pandas as pd
from numpy.testing import assert_allclose, assert_equal
import statsmodels.api as sm
def test_log_riskratio(self):
    assert_allclose(self.tbl_obj.log_riskratio, self.log_riskratio)