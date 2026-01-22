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
def test_expect(self):

    def func(x):
        return x
    gm = stats.gamma(a=2, loc=3, scale=4)
    with np.errstate(invalid='ignore', divide='ignore'):
        gm_val = gm.expect(func, lb=1, ub=2, conditional=True)
        gamma_val = stats.gamma.expect(func, args=(2,), loc=3, scale=4, lb=1, ub=2, conditional=True)
    assert_allclose(gm_val, gamma_val)
    p = stats.poisson(3, loc=4)
    p_val = p.expect(func)
    poisson_val = stats.poisson.expect(func, args=(3,), loc=4)
    assert_allclose(p_val, poisson_val)