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
def test_skewcauchy_R(self):
    np.random.seed(0)
    a = np.random.rand(10) * 2 - 1
    x = np.random.rand(10) * 10 - 5
    pdf = [0.03947397521733391, 0.3058297140499032, 0.24140158118994162, 0.019585772402693054, 0.021436553695989482, 0.00909817103867518, 0.01658423410016873, 0.07108328803039413, 0.10325004594145452, 0.013110230778426242]
    cdf = [0.8742667771821375, 0.3755646891078088, 0.5944209649653807, 0.913046598508902, 0.09631964100300605, 0.03829624330921733, 0.08245240578402535, 0.7205706294551039, 0.6282641585251545, 0.9501130846389829]
    assert_allclose(stats.skewcauchy.pdf(x, a), pdf)
    assert_allclose(stats.skewcauchy.cdf(x, a), cdf)
    assert_allclose(stats.skewcauchy.ppf(cdf, a), x)