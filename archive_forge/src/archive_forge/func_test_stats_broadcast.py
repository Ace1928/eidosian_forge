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
def test_stats_broadcast(self):
    dfn = np.array([[3], [11]])
    dfd = np.array([11, 12])
    m, v, s, k = stats.f.stats(dfn=dfn, dfd=dfd, moments='mvsk')
    m2 = [dfd / (dfd - 2)] * 2
    assert_allclose(m, m2)
    v2 = 2 * dfd ** 2 * (dfn + dfd - 2) / dfn / (dfd - 2) ** 2 / (dfd - 4)
    assert_allclose(v, v2)
    s2 = (2 * dfn + dfd - 2) * np.sqrt(8 * (dfd - 4)) / ((dfd - 6) * np.sqrt(dfn * (dfn + dfd - 2)))
    assert_allclose(s, s2)
    k2num = 12 * (dfn * (5 * dfd - 22) * (dfn + dfd - 2) + (dfd - 4) * (dfd - 2) ** 2)
    k2den = dfn * (dfd - 6) * (dfd - 8) * (dfn + dfd - 2)
    k2 = k2num / k2den
    assert_allclose(k, k2)