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
def test_gh_11299_rvs(self):
    low = [-10, 10, -np.inf, -5, -np.inf, -np.inf, -45, -45, 40, -10, 40]
    high = [-5, 11, 5, np.inf, 40, -40, 40, -40, 45, np.inf, np.inf]
    x = stats.truncnorm.rvs(low, high, size=(5, len(low)))
    assert np.shape(x) == (5, len(low))
    assert_(np.all(low <= x.min(axis=0)))
    assert_(np.all(x.max(axis=0) <= high))