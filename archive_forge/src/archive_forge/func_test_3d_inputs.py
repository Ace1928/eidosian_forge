import os
import re
import warnings
from collections import namedtuple
from itertools import product
import hypothesis.extra.numpy as npst
import hypothesis
import contextlib
from numpy.testing import (assert_, assert_equal,
import pytest
from pytest import raises as assert_raises
import numpy.ma.testutils as mat
from numpy import array, arange, float32, float64, power
import numpy as np
import scipy.stats as stats
import scipy.stats.mstats as mstats
import scipy.stats._mstats_basic as mstats_basic
from scipy.stats._ksstats import kolmogn
from scipy.special._testutils import FuncData
from scipy.special import binom
from scipy import optimize
from .common_tests import check_named_results
from scipy.spatial.distance import cdist
from scipy.stats._axis_nan_policy import _broadcast_concatenate
from scipy.stats._stats_py import _permutation_distribution_t
from scipy._lib._util import AxisError
def test_3d_inputs(self):
    a = 1 / np.arange(1.0, 4 * 5 * 7 + 1).reshape(4, 5, 7)
    b = 2 / np.arange(1.0, 4 * 8 * 7 + 1).reshape(4, 8, 7)
    c = np.cos(1 / np.arange(1.0, 4 * 4 * 7 + 1).reshape(4, 4, 7))
    f, p = stats.f_oneway(a, b, c, axis=1)
    assert f.shape == (4, 7)
    assert p.shape == (4, 7)
    for i in range(a.shape[0]):
        for j in range(a.shape[2]):
            fij, pij = stats.f_oneway(a[i, :, j], b[i, :, j], c[i, :, j])
            assert_allclose(fij, f[i, j])
            assert_allclose(pij, p[i, j])