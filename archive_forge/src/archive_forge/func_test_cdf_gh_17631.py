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
@pytest.mark.parametrize('x, a, b, expected', cdf_vals + [(10000000000.0, 1.5, 1.5, 0.9999999999999983), (10000000000.0, 0.05, 0.1, 0.9664184367890859), (1e+22, 0.05, 0.1, 0.9978811466052919)])
def test_cdf_gh_17631(self, x, a, b, expected):
    assert_allclose(stats.betaprime.cdf(x, a, b), expected, rtol=1e-14)