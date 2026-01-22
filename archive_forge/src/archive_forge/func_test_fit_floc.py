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
@pytest.mark.parametrize('rho,gamma', [pytest.param(36.545206797050334, 2.4952, marks=pytest.mark.slow), pytest.param(38.55107913669065, 2.085, marks=pytest.mark.xslow), pytest.param(96292.3076923077, 0.0013, marks=pytest.mark.xslow)])
def test_fit_floc(self, rho, gamma):
    """Tests fit for cases where floc is set.

        `rel_breitwigner` has special handling for these cases.
        """
    seed = 6936804688480013683
    rng = np.random.default_rng(seed)
    data = stats.rel_breitwigner.rvs(rho, scale=gamma, size=1000, random_state=rng)
    fit = stats.rel_breitwigner.fit(data, floc=0)
    assert_allclose((fit[0], fit[2]), (rho, gamma), rtol=0.2)
    assert fit[1] == 0
    fit = stats.rel_breitwigner.fit(data, floc=0, fscale=gamma)
    assert_allclose(fit[0], rho, rtol=0.01)
    assert (fit[1], fit[2]) == (0, gamma)