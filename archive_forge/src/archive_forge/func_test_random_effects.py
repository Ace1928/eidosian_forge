from statsmodels.compat.platform import PLATFORM_OSX
import os
import csv
import warnings
import numpy as np
import pandas as pd
from scipy import sparse
import pytest
from statsmodels.regression.mixed_linear_model import (
from numpy.testing import (assert_almost_equal, assert_equal, assert_allclose,
from statsmodels.base import _penalties as penalties
import statsmodels.tools.numdiff as nd
from .results import lme_r_results
def test_random_effects():
    np.random.seed(23429)
    ngrp = 100
    gsize = 10
    rsd = 2
    gsd = 3
    mn = gsd * np.random.normal(size=ngrp)
    gmn = np.kron(mn, np.ones(gsize))
    y = gmn + rsd * np.random.normal(size=ngrp * gsize)
    gr = np.kron(np.arange(ngrp), np.ones(gsize))
    x = np.ones(ngrp * gsize)
    model = MixedLM(y, x, groups=gr)
    result = model.fit()
    re = result.random_effects
    assert_(isinstance(re, dict))
    assert_(len(re) == ngrp)
    assert_(isinstance(re[0], pd.Series))
    assert_(len(re[0]) == 1)
    model = MixedLM(y, x, exog_re=x, groups=gr)
    result = model.fit()
    re = result.random_effects
    assert_(isinstance(re, dict))
    assert_(len(re) == ngrp)
    assert_(isinstance(re[0], pd.Series))
    assert_(len(re[0]) == 1)
    xr = np.random.normal(size=(ngrp * gsize, 2))
    xr[:, 0] = 1
    qp = np.linspace(-1, 1, gsize)
    xr[:, 1] = np.kron(np.ones(ngrp), qp)
    model = MixedLM(y, x, exog_re=xr, groups=gr)
    result = model.fit()
    re = result.random_effects
    assert_(isinstance(re, dict))
    assert_(len(re) == ngrp)
    assert_(isinstance(re[0], pd.Series))
    assert_(len(re[0]) == 2)