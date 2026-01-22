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
def compare_to_reference(sp, sp_dict, decimal=(12, 12)):
    assert_allclose(sp[0], sp_dict['statistic'], atol=10 ** (-decimal[0]), rtol=10 ** (-decimal[0]))
    assert_allclose(sp[1], sp_dict['pvalue'], atol=10 ** (-decimal[1]), rtol=10 ** (-decimal[0]))