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
def test_cdf_sf(self):
    mu = [0.000417022005, 0.00720324493, 1.14374817e-06, 0.00302332573, 0.00146755891]
    expected = [1, 1, 1, 1, 1]
    actual = stats.invgauss.cdf(0.4, mu=mu)
    assert_equal(expected, actual)
    cdf_actual = stats.invgauss.cdf(0.001, mu=1.05)
    assert_allclose(cdf_actual, 4.65246506892667e-219)
    sf_actual = stats.invgauss.sf(110, mu=1.05)
    assert_allclose(sf_actual, 4.12851625944048e-25)
    actual = stats.invgauss.cdf(9e-05, 0.0001)
    assert_allclose(actual, 2.9458022894924e-26)
    actual = stats.invgauss.cdf(0.000102, 0.0001)
    assert_allclose(actual, 0.976445540507925)