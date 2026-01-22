import os
import numpy as np
import numpy.testing as npt
from numpy.testing import assert_allclose, assert_equal
import pytest
from scipy import stats
from scipy.optimize import differential_evolution
from .test_continuous_basic import distcont
from scipy.stats._distn_infrastructure import FitError
from scipy.stats._distr_params import distdiscrete
from scipy.stats import goodness_of_fit
@pytest.mark.parametrize('dist_name', cases_test_fit_mse())
def test_basic_fit_mse(self, dist_name):
    self.basic_fit_test(dist_name, 'mse')