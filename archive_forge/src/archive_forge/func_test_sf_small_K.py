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
@pytest.mark.parametrize('x, K, scale, expected', [[10, 0.01, 1, 8.474702916146657e-24], [2, 0.005, 1, 0.02302280664231312], [5, 0.005, 0.5, 8.024820681931086e-24], [10, 0.005, 0.5, 3.0603340062892486e-89], [20, 0.005, 0.5, 0.0], [-3, 0.001, 1, 0.9986545205566117]])
def test_sf_small_K(self, x, K, scale, expected):
    p = stats.exponnorm.sf(x, K, scale=scale)
    if expected == 0.0:
        assert p == 0.0
    else:
        assert_allclose(p, expected, rtol=5e-13)