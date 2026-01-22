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
def test_moments_r(self):
    vals_R = [2.36848366948115, 8.4739346779246, 37.8870502710066, 205.76608511485]
    lmbda, alpha, beta = (2, 2, 1)
    mu, delta = (0.5, 1.5)
    args = (lmbda, alpha * delta, beta * delta)
    vals_us = [stats.genhyperbolic(*args, loc=mu, scale=delta).moment(i) for i in range(1, 5)]
    assert_allclose(vals_us, vals_R, atol=0, rtol=1e-13)