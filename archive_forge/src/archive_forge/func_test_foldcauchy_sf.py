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
@pytest.mark.parametrize('x, c, expected', [(2, 1, 0.35241638234956674), (2, 2, 0.5779791303773694), (10000000000000.0, 1, 6.366197723675813e-14), (2e+16, 1, 3.183098861837907e-17), (10000000000000.0, 200000000000.0, 6.368745221764519e-14), (0.125, 200, 0.999998010612169)])
def test_foldcauchy_sf(x, c, expected):
    sf = stats.foldcauchy.sf(x, c)
    assert_allclose(sf, expected, 2e-15)