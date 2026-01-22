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
@pytest.mark.parametrize('a, c, ref, tol', [(1500000.0, 1, 8.529426144018633, 1e-15), (1e+30, 1, 35.95771492811536, 1e-15), (1e+100, 1, 116.54819318290696, 1e-15), (3000.0, 1, 5.422011196659015, 1e-13), (3000000.0, -1e+100, -236.29663213396054, 1e-15), (3e+60, 1e-100, 1.3925371786831085e+102, 1e-15)])
def test_gengamma_extreme_entropy(a, c, ref, tol):
    assert_allclose(stats.gengamma.entropy(a, c), ref, rtol=tol)