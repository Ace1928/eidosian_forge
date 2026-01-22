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
@pytest.mark.slow
def test_regularized(self):
    np.random.seed(3453)
    exog = np.random.normal(size=(400, 5))
    groups = np.kron(np.arange(100), np.ones(4))
    expected_endog = exog[:, 0] - exog[:, 2]
    endog = expected_endog + np.kron(np.random.normal(size=100), np.ones(4)) + np.random.normal(size=400)
    md = MixedLM(endog, exog, groups)
    mdf1 = md.fit_regularized(alpha=1.0)
    mdf1.summary()
    md = MixedLM(endog, exog, groups)
    mdf2 = md.fit_regularized(alpha=10 * np.ones(5))
    mdf2.summary()
    pen = penalties.L2()
    mdf3 = md.fit_regularized(method=pen, alpha=0.0)
    mdf3.summary()
    pen = penalties.L2()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mdf4 = md.fit_regularized(method=pen, alpha=10.0)
    mdf4.summary()
    pen = penalties.PseudoHuber(0.3)
    mdf5 = md.fit_regularized(method=pen, alpha=1.0)
    mdf5.summary()