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
def test_acorr_breusch_godfrey_exogs(self):
    data = sunspots.load_pandas().data['SUNACTIVITY']
    res = ARIMA(data, order=(1, 0, 0), trend='n').fit()
    smsdia.acorr_breusch_godfrey(res, nlags=1)