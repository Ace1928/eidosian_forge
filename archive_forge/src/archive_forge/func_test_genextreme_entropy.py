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
def test_genextreme_entropy():
    euler_gamma = 0.5772156649015329
    h = stats.genextreme.entropy(-1.0)
    assert_allclose(h, 2 * euler_gamma + 1, rtol=1e-14)
    h = stats.genextreme.entropy(0)
    assert_allclose(h, euler_gamma + 1, rtol=1e-14)
    h = stats.genextreme.entropy(1.0)
    assert_equal(h, 1)
    h = stats.genextreme.entropy(-2.0, scale=10)
    assert_allclose(h, euler_gamma * 3 + np.log(10) + 1, rtol=1e-14)
    h = stats.genextreme.entropy(10)
    assert_allclose(h, -9 * euler_gamma + 1, rtol=1e-14)
    h = stats.genextreme.entropy(-10)
    assert_allclose(h, 11 * euler_gamma + 1, rtol=1e-14)