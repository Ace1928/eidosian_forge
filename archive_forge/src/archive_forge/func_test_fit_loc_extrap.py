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
def test_fit_loc_extrap(self):
    x = [1, 1, 3, 3, 10, 10, 10, 30, 30, 140, 140]
    alpha1, beta1, loc1, scale1 = stats.levy_stable._fitstart(x)
    assert alpha1 < 1, f'Expected alpha < 1, got {alpha1}'
    assert loc1 < min(x), f'Expected loc < {min(x)}, got {loc1}'
    x2 = [1, 1, 3, 3, 10, 10, 10, 30, 30, 130, 130]
    alpha2, beta2, loc2, scale2 = stats.levy_stable._fitstart(x2)
    assert alpha2 > 1, f'Expected alpha > 1, got {alpha2}'
    assert loc2 > max(x2), f'Expected loc > {max(x2)}, got {loc2}'