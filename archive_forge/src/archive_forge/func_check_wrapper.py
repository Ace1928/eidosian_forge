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
def check_wrapper(results):
    assert_(isinstance(results.params, pd.Series))
    assert_(isinstance(results.fittedvalues, pd.Series))
    assert_(isinstance(results.resid, pd.Series))
    assert_(isinstance(results.centered_resid, pd.Series))
    assert_(isinstance(results._results.params, np.ndarray))
    assert_(isinstance(results._results.fittedvalues, np.ndarray))
    assert_(isinstance(results._results.resid, np.ndarray))
    assert_(isinstance(results._results.centered_resid, np.ndarray))