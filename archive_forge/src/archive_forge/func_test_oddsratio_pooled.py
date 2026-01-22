import os
import warnings
import numpy as np
import statsmodels.stats.contingency_tables as ctab
import pandas as pd
from numpy.testing import assert_allclose, assert_equal
import statsmodels.api as sm
def test_oddsratio_pooled(self):
    assert_allclose(self.rslt.oddsratio_pooled, self.oddsratio_pooled, rtol=0.0001, atol=0.0001)