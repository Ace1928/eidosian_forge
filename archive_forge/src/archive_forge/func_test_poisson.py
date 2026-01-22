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
def test_poisson(self):
    prob_bounds = stats.poisson.expect(lambda x: 1, args=(2,), lb=3, conditional=False)
    prob_b_true = 1 - stats.poisson.cdf(2, 2)
    assert_almost_equal(prob_bounds, prob_b_true, decimal=14)
    prob_lb = stats.poisson.expect(lambda x: 1, args=(2,), lb=2, conditional=True)
    assert_almost_equal(prob_lb, 1, decimal=14)