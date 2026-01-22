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
@pytest.mark.parametrize('rvs_shape', [0.1, 2])
@pytest.mark.parametrize('rvs_loc', [-2, 0, 2])
@pytest.mark.parametrize('rvs_scale', [0.2, 1, 5])
@pytest.mark.parametrize('fix_shape, fix_loc, fix_scale', [e for e in product((False, True), repeat=3) if False in e])
@np.errstate(invalid='ignore')
def test_fit_MLE_comp_optimizer(self, rvs_shape, rvs_loc, rvs_scale, fix_shape, fix_loc, fix_scale, rng):
    data = stats.lognorm.rvs(size=100, s=rvs_shape, scale=rvs_scale, loc=rvs_loc, random_state=rng)
    kwds = {}
    if fix_shape:
        kwds['f0'] = rvs_shape
    if fix_loc:
        kwds['floc'] = rvs_loc
    if fix_scale:
        kwds['fscale'] = rvs_scale
    _assert_less_or_close_loglike(stats.lognorm, data, **kwds)