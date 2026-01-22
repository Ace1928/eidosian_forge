import numpy as np
from numpy.testing import assert_allclose, assert_equal
from statsmodels.regression.linear_model import OLS
from statsmodels.stats._delta_method import NonlinearDeltaCov
Unit tests for NonlinearDeltaCov and LikelihoodResults._get_wald_nonlinear
Created on Sun Mar 01 01:05:35 2015

Author: Josef Perktold
License: BSD-3

