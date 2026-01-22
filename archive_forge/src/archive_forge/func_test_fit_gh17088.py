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
def test_fit_gh17088(self):
    rng = np.random.default_rng(456)
    loc, scale, size = (50, 600, 500)
    rvs = stats.rayleigh.rvs(loc, scale, size=size, random_state=rng)
    loc_fit, _ = stats.rayleigh.fit(rvs)
    assert loc_fit < np.min(rvs)
    loc_fit, scale_fit = stats.rayleigh.fit(rvs, fscale=scale)
    assert loc_fit < np.min(rvs)
    assert scale_fit == scale