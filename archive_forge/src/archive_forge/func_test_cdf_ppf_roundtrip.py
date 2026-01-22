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
def test_cdf_ppf_roundtrip(self):
    q = np.r_[np.logspace(1e-12, 0.01, base=0.1), np.linspace(0.01, 1, 30, endpoint=False), 1.0 - np.logspace(1e-12, 0.01, base=0.1)]
    for c in [1e-08, -1e-18, 1e-15, -1e-15]:
        assert_allclose(stats.genpareto.cdf(stats.genpareto.ppf(q, c), c), q, atol=1e-15)