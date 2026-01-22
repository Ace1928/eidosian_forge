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
from scipy.optimize import root, fmin
from itertools import product
@pytest.mark.parametrize('loc', [-0.5 * np.pi, 0, np.pi])
@pytest.mark.parametrize('kappa_tol', [(0.1, 0.05), (100.0, 0.01), (100000.0, 0.01)])
def test_vonmises_fit_all(self, kappa_tol, loc):
    rng = np.random.default_rng(6762668991392531563)
    kappa, tol = kappa_tol
    data = stats.vonmises(loc=loc, kappa=kappa).rvs(100000, random_state=rng)
    kappa_fit, loc_fit, scale_fit = stats.vonmises.fit(data)
    assert scale_fit == 1
    loc_vec = np.array([np.cos(loc), np.sin(loc)])
    loc_fit_vec = np.array([np.cos(loc_fit), np.sin(loc_fit)])
    angle = np.arccos(loc_vec.dot(loc_fit_vec))
    assert_allclose(angle, 0, atol=tol, rtol=0)
    assert_allclose(kappa, kappa_fit, rtol=tol)