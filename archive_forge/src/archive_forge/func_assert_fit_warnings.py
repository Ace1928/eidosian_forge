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
def assert_fit_warnings(dist):
    param = ['floc', 'fscale']
    if dist.shapes:
        nshapes = len(dist.shapes.split(','))
        param += ['f0', 'f1', 'f2'][:nshapes]
    all_fixed = dict(zip(param, np.arange(len(param))))
    data = [1, 2, 3]
    with pytest.raises(RuntimeError, match='All parameters fixed. There is nothing to optimize.'):
        dist.fit(data, **all_fixed)
    with pytest.raises(ValueError, match='The data contains non-finite values'):
        dist.fit([np.nan])
    with pytest.raises(ValueError, match='The data contains non-finite values'):
        dist.fit([np.inf])
    with pytest.raises(TypeError, match='Unknown keyword arguments:'):
        dist.fit(data, extra_keyword=2)
    with pytest.raises(TypeError, match='Too many positional arguments.'):
        dist.fit(data, *[1] * (len(param) - 1))