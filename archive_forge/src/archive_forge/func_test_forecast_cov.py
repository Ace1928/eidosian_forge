from statsmodels.compat.pandas import QUARTER_END, assert_index_equal
from statsmodels.compat.python import lrange
from io import BytesIO, StringIO
import os
import sys
import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal, assert_equal
import pandas as pd
import pytest
from statsmodels.datasets import macrodata
import statsmodels.tools.data as data_util
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.tsa.base.datetools import dates_from_str
import statsmodels.tsa.vector_ar.util as util
from statsmodels.tsa.vector_ar.var_model import VAR, var_acf
def test_forecast_cov(self):
    res = self.res0
    covfc1 = res.forecast_cov(3)
    assert_allclose(covfc1, res.mse(3), rtol=1e-13)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        covfc2 = res.forecast_cov(3, method='auto')
    assert_allclose(covfc2, covfc1, rtol=0.05)
    res_covfc2 = np.array([[[9.45802013, 4.94142038, 37.1999646], [4.94142038, 7.09273624, 5.66215089], [37.1999646, 5.66215089, 259.61275869]], [[11.30364479, 5.72569141, 49.28744123], [5.72569141, 7.409761, 10.98164091], [49.28744123, 10.98164091, 336.4484723]], [[12.36188803, 6.44426905, 53.54588026], [6.44426905, 7.88850029, 13.96382545], [53.54588026, 13.96382545, 352.19564327]]])
    assert_allclose(covfc2, res_covfc2, atol=1e-06)