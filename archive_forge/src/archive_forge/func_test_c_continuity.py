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
def test_c_continuity(self):
    x = np.linspace(0, 10, 30)
    for c in [0, -1]:
        pdf0 = stats.genpareto.pdf(x, c)
        for dc in [1e-14, -1e-14]:
            pdfc = stats.genpareto.pdf(x, c + dc)
            assert_allclose(pdf0, pdfc, atol=1e-12)
        cdf0 = stats.genpareto.cdf(x, c)
        for dc in [1e-14, 1e-14]:
            cdfc = stats.genpareto.cdf(x, c + dc)
            assert_allclose(cdf0, cdfc, atol=1e-12)