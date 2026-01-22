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
def test_harvey_collier(self):
    harvey_collier = dict(statistic=0.494432160939874, pvalue=0.6215491310408242, parameters=198, distr='t')
    hc = smsdia.linear_harvey_collier(self.res)
    compare_to_reference(hc, harvey_collier, decimal=(12, 12))
    hc_skip = smsdia.linear_harvey_collier(self.res, skip=20)
    assert not np.allclose(hc[0], hc_skip[0])