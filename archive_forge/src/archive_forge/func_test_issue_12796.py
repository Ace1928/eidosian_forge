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
@pytest.mark.skipif(MACOS_INTEL, reason='Overflow, see gh-14901')
def test_issue_12796(self):
    alpha_2 = 5e-06
    count_ = np.arange(1, 20)
    nobs = 100000
    q, a, b = (1 - alpha_2, count_ + 1, nobs - count_)
    inv = stats.beta.ppf(q, a, b)
    res = stats.beta.cdf(inv, a, b)
    assert_allclose(res, 1 - alpha_2)