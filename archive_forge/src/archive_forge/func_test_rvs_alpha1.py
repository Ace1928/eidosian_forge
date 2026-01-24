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
@pytest.mark.slow
@pytest.mark.parametrize('beta', [0.5, 1])
def test_rvs_alpha1(self, beta):
    """Additional test cases for rvs for alpha equal to 1."""
    np.random.seed(989284321)
    alpha = 1.0
    loc = 0.5
    scale = 1.5
    x = stats.levy_stable.rvs(alpha, beta, loc=loc, scale=scale, size=5000)
    stat, p = stats.kstest(x, 'levy_stable', args=(alpha, beta, loc, scale))
    assert p > 0.01