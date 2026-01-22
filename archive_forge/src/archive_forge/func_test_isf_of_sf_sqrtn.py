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
def test_isf_of_sf_sqrtn(self):
    x = np.linspace(0, 1, 11)
    for n in [1, 2, 3, 10, 100, 1000]:
        xn = (x / np.sqrt(n))[x > 0.5 / n]
        vals_sf = stats.kstwo.sf(xn, n)
        cond = (0 < vals_sf) & (vals_sf < 0.95)
        vals = stats.kstwo.isf(vals_sf, n)
        assert_allclose(vals[cond], xn[cond])