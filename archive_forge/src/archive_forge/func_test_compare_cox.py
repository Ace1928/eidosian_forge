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
def test_compare_cox(self):
    res = self.res
    res2 = self.res2
    coxtest = [('fitted(M1) ~ M2', -0.782030488930356, 0.599696502782265, -1.304043770977755, 0.1922186587840554, ' '), ('fitted(M2) ~ M1', -2.248817107408537, 0.392656854330139, -5.727181590258883, 1.021128495098556e-08, '***')]
    ct1 = smsdia.compare_cox(res, res2)
    assert_almost_equal(ct1, coxtest[0][3:5], decimal=12)
    ct2 = smsdia.compare_cox(res2, res)
    assert_almost_equal(ct2, coxtest[1][3:5], decimal=12)
    _, _, store = smsdia.compare_cox(res, res2, store=True)
    assert isinstance(store, smsdia.ResultsStore)