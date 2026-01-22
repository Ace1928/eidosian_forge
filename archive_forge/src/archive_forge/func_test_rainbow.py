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
def test_rainbow(self):
    raintest = dict(statistic=0.6809600116739604, pvalue=0.971832843583418, parameters=(101, 98), distr='f')
    raintest_fraction_04 = dict(statistic=0.565551237772662, pvalue=0.997592305968473, parameters=(122, 77), distr='f')
    rb = smsdia.linear_rainbow(self.res)
    compare_to_reference(rb, raintest, decimal=(12, 12))
    rb = smsdia.linear_rainbow(self.res, frac=0.4)
    compare_to_reference(rb, raintest_fraction_04, decimal=(12, 12))