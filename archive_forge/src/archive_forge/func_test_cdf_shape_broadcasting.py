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
def test_cdf_shape_broadcasting(self):
    c = np.array([[0.03, 0.25], [0.5, 0.75]])
    x = np.array([[1.0], [4.0]])
    p = stats.wrapcauchy.cdf(x, c)
    assert p.shape == (2, 2)
    scalar_values = [stats.wrapcauchy.cdf(x1, c1) for x1, c1 in np.nditer((x, c))]
    assert_allclose(p.ravel(), scalar_values, rtol=1e-13)