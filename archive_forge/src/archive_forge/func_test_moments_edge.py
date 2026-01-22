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
def test_moments_edge(self):
    c, d = (2, 2)
    mean = np.pi / 4
    var = 1 - np.pi ** 2 / 16
    skew = np.pi ** 3 / (32 * var ** 1.5)
    kurtosis = np.nan
    ref = [mean, var, skew, kurtosis]
    res = stats.burr12(c, d).stats('mvsk')
    assert_allclose(res, ref, rtol=1e-14)