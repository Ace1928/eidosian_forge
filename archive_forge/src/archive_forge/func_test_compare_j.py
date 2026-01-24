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
def test_compare_j(self):
    res = self.res
    res2 = self.res2
    jtest = [('M1 + fitted(M2)', 1.591505670785873, 0.7384552861695823, 2.15518217635237, 0.03235457252531445, '*'), ('M2 + fitted(M1)', 1.305689283016899, 0.4808385176653064, 2.715438978051544, 0.007203854534057954, '**')]
    jt1 = smsdia.compare_j(res2, res)
    assert_almost_equal(jt1, jtest[0][3:5], decimal=12)
    jt2 = smsdia.compare_j(res, res2)
    assert_almost_equal(jt2, jtest[1][3:5], decimal=12)