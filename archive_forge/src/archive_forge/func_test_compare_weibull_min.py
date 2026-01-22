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
def test_compare_weibull_min(self):
    x = 1.5
    c = 2.0
    a = 0.0
    b = np.inf
    scale = 3.0
    p = stats.weibull_min.pdf(x, c, scale=scale)
    p_trunc = stats.truncweibull_min.pdf(x, c, a, b, scale=scale)
    assert_allclose(p, p_trunc)
    lp = stats.weibull_min.logpdf(x, c, scale=scale)
    lp_trunc = stats.truncweibull_min.logpdf(x, c, a, b, scale=scale)
    assert_allclose(lp, lp_trunc)
    cdf = stats.weibull_min.cdf(x, c, scale=scale)
    cdf_trunc = stats.truncweibull_min.cdf(x, c, a, b, scale=scale)
    assert_allclose(cdf, cdf_trunc)
    lc = stats.weibull_min.logcdf(x, c, scale=scale)
    lc_trunc = stats.truncweibull_min.logcdf(x, c, a, b, scale=scale)
    assert_allclose(lc, lc_trunc)
    s = stats.weibull_min.sf(x, c, scale=scale)
    s_trunc = stats.truncweibull_min.sf(x, c, a, b, scale=scale)
    assert_allclose(s, s_trunc)
    ls = stats.weibull_min.logsf(x, c, scale=scale)
    ls_trunc = stats.truncweibull_min.logsf(x, c, a, b, scale=scale)
    assert_allclose(ls, ls_trunc)
    s = stats.truncweibull_min.sf(30, 2, a, b, scale=3)
    assert_allclose(s, np.exp(-100))
    ls = stats.truncweibull_min.logsf(30, 2, a, b, scale=3)
    assert_allclose(ls, -100)