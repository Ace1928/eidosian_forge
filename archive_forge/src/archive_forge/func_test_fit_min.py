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
def test_fit_min(self):
    rng = np.random.default_rng(5985959307161735394)
    c, loc, scale = (2, 3.5, 0.5)
    dist = stats.weibull_min(c, loc, scale)
    rvs = dist.rvs(size=100, random_state=rng)
    c2, loc2, scale2 = stats.weibull_min.fit(rvs, 1.5, floc=3)
    c3, loc3, scale3 = stats.weibull_min.fit(rvs, 1.6, floc=3)
    assert loc2 == loc3 == 3
    assert c2 != c3
    c4, loc4, scale4 = stats.weibull_min.fit(rvs, 3, fscale=3, method='mm')
    assert scale4 == 3
    dist4 = stats.weibull_min(c4, loc4, scale4)
    res = dist4.stats(moments='ms')
    ref = (np.mean(rvs), stats.skew(rvs))
    assert_allclose(res, ref)