import json
import os
import numpy as np
from numpy.testing import (
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest
from statsmodels.datasets import macrodata, sunspots
from statsmodels.regression.linear_model import OLS
import statsmodels.stats.diagnostic as smsdia
import statsmodels.stats.outliers_influence as oi
import statsmodels.stats.sandwich_covariance as sw
from statsmodels.tools.tools import Bunch, add_constant
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
def test_het_arch(self):
    archtest_4 = dict(statistic=3.43473400836259, pvalue=0.487871315392619, parameters=(4,), distr='chi2')
    archtest_12 = dict(statistic=8.648320999014171, pvalue=0.732638635007718, parameters=(12,), distr='chi2')
    at4 = smsdia.het_arch(self.res.resid, nlags=4)
    at12 = smsdia.het_arch(self.res.resid, nlags=12)
    compare_to_reference(at4[:2], archtest_4, decimal=(12, 13))
    compare_to_reference(at12[:2], archtest_12, decimal=(12, 13))