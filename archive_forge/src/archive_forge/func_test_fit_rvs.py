import warnings
import re
import sys
import pickle
from pathlib import Path
import os
import json
import platform
from numpy.testing import (assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
import numpy
import numpy as np
from numpy import typecodes, array
from numpy.lib.recfunctions import rec_append_fields
from scipy import special
from scipy._lib._util import check_random_state
from scipy.integrate import (IntegrationWarning, quad, trapezoid,
import scipy.stats as stats
from scipy.stats._distn_infrastructure import argsreduce
import scipy.stats.distributions
from scipy.special import xlogy, polygamma, entr
from scipy.stats._distr_params import distcont, invdistcont
from .test_discrete_basic import distdiscrete, invdistdiscrete
from scipy.stats._continuous_distns import FitDataError, _argus_phi
from scipy.optimize import root, fmin, differential_evolution
from itertools import product
@pytest.mark.xfail(reason='Unknown problem with fitstart.')
@pytest.mark.parametrize('alpha,beta,delta,gamma', [(1.5, 0.4, 2, 3), (1.0, 0.4, 2, 3)])
@pytest.mark.parametrize('parametrization', ['S0', 'S1'])
def test_fit_rvs(self, alpha, beta, delta, gamma, parametrization):
    """Test that fit agrees with rvs for each parametrization."""
    stats.levy_stable.parametrization = parametrization
    data = stats.levy_stable.rvs(alpha, beta, loc=delta, scale=gamma, size=10000, random_state=1234)
    fit = stats.levy_stable._fitstart(data)
    alpha_obs, beta_obs, delta_obs, gamma_obs = fit
    assert_allclose([alpha, beta, delta, gamma], [alpha_obs, beta_obs, delta_obs, gamma_obs], rtol=0.01)