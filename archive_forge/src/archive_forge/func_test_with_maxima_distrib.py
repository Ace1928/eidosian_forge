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
def test_with_maxima_distrib(self):
    x = 1.5
    a = 2.0
    b = 3.0
    p = stats.weibull_min.pdf(x, a, scale=b)
    assert_allclose(p, np.exp(-0.25) / 3)
    lp = stats.weibull_min.logpdf(x, a, scale=b)
    assert_allclose(lp, -0.25 - np.log(3))
    c = stats.weibull_min.cdf(x, a, scale=b)
    assert_allclose(c, -special.expm1(-0.25))
    lc = stats.weibull_min.logcdf(x, a, scale=b)
    assert_allclose(lc, np.log(-special.expm1(-0.25)))
    s = stats.weibull_min.sf(x, a, scale=b)
    assert_allclose(s, np.exp(-0.25))
    ls = stats.weibull_min.logsf(x, a, scale=b)
    assert_allclose(ls, -0.25)
    s = stats.weibull_min.sf(30, 2, scale=3)
    assert_allclose(s, np.exp(-100))
    ls = stats.weibull_min.logsf(30, 2, scale=3)
    assert_allclose(ls, -100)
    x = -1.5
    p = stats.weibull_max.pdf(x, a, scale=b)
    assert_allclose(p, np.exp(-0.25) / 3)
    lp = stats.weibull_max.logpdf(x, a, scale=b)
    assert_allclose(lp, -0.25 - np.log(3))
    c = stats.weibull_max.cdf(x, a, scale=b)
    assert_allclose(c, np.exp(-0.25))
    lc = stats.weibull_max.logcdf(x, a, scale=b)
    assert_allclose(lc, -0.25)
    s = stats.weibull_max.sf(x, a, scale=b)
    assert_allclose(s, -special.expm1(-0.25))
    ls = stats.weibull_max.logsf(x, a, scale=b)
    assert_allclose(ls, np.log(-special.expm1(-0.25)))
    s = stats.weibull_max.sf(-1e-09, 2, scale=3)
    assert_allclose(s, -special.expm1(-1 / 9000000000000000000))
    ls = stats.weibull_max.logsf(-1e-09, 2, scale=3)
    assert_allclose(ls, np.log(-special.expm1(-1 / 9000000000000000000)))