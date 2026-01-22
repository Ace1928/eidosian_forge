from statsmodels.compat import lrange
import os
import numpy as np
import pytest
from numpy.testing import (assert_almost_equal, assert_equal, assert_allclose,
import statsmodels.genmod.generalized_estimating_equations as gee
import statsmodels.tools as tools
import statsmodels.regression.linear_model as lm
from statsmodels.genmod import families
from statsmodels.genmod import cov_struct
import statsmodels.discrete.discrete_model as discrete
import pandas as pd
from scipy.stats.distributions import norm
import warnings
def simple_qic_data(fam):
    y = np.r_[0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0]
    x1 = np.r_[0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0]
    x2 = np.r_[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    g = np.r_[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
    x1 = x1[:, None]
    x2 = x2[:, None]
    return (y, x1, x2, g)