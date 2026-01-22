from statsmodels.compat.python import lmap
import numpy as np
from numpy.testing import assert_almost_equal
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
import statsmodels.stats.sandwich_covariance as sw
Test for panel robust covariance estimators after pooled ols
this follows the example from xtscc paper/help

Created on Tue May 22 20:27:57 2012

Author: Josef Perktold
