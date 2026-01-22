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
def test_acf_2_lags(self):
    c = np.zeros((2, 2, 2))
    c[0] = np.array([[0.2, 0.1], [0.15, 0.15]])
    c[1] = np.array([[0.1, 0.9], [0, 0.1]])
    acf = var_acf(c, np.eye(2), 3)
    gamma = np.zeros((6, 6))
    gamma[:2, :2] = acf[0]
    gamma[2:4, 2:4] = acf[0]
    gamma[4:6, 4:6] = acf[0]
    gamma[2:4, :2] = acf[1].T
    gamma[4:, :2] = acf[2].T
    gamma[:2, 2:4] = acf[1]
    gamma[:2, 4:] = acf[2]
    recovered = np.dot(gamma[:2, 2:], np.linalg.inv(gamma[:4, :4]))
    recovered = [recovered[:, 2 * i:2 * (i + 1)] for i in range(2)]
    recovered = np.array(recovered)
    assert_allclose(recovered, c, atol=1e-07)