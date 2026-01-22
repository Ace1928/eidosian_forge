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
def test_vonmises_fit_shape(self):
    rng = np.random.default_rng(6762668991392531563)
    loc = 0.25 * np.pi
    kappa = 10
    data = stats.vonmises(loc=loc, kappa=kappa).rvs(100000, random_state=rng)
    kappa_fit, loc_fit, scale_fit = stats.vonmises.fit(data, floc=loc)
    assert loc_fit == loc
    assert scale_fit == 1
    assert_allclose(kappa, kappa_fit, rtol=0.01)