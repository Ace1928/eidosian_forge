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
def test_pastes_vcomp(self):
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    rdir = os.path.join(cur_dir, 'results')
    fname = os.path.join(rdir, 'pastes.csv')
    data = pd.read_csv(fname)
    vcf = {'cask': '0 + cask'}
    model = MixedLM.from_formula('strength ~ 1', groups='batch', re_formula='1', vc_formula=vcf, data=data)
    result = model.fit()
    assert_allclose(result.fe_params.iloc[0], 60.0533, rtol=0.001)
    assert_allclose(result.bse.iloc[0], 0.6769, rtol=0.001)
    assert_allclose(result.cov_re.iloc[0, 0], 1.657, rtol=0.001)
    assert_allclose(result.scale, 0.678, rtol=0.001)
    assert_allclose(result.llf, -123.49, rtol=0.1)
    assert_equal(result.aic, np.nan)
    assert_equal(result.bic, np.nan)
    resid = np.r_[0.17133538, -0.02866462, -1.08662875, 1.11337125, -0.12093607]
    assert_allclose(result.resid[0:5], resid, rtol=0.001)
    fit = np.r_[62.62866, 62.62866, 61.18663, 61.18663, 62.82094]
    assert_allclose(result.fittedvalues[0:5], fit, rtol=0.0001)
    model = MixedLM.from_formula('strength ~ 1', groups='batch', re_formula='1', vc_formula=vcf, data=data)
    result = model.fit(reml=False)
    assert_allclose(result.fe_params.iloc[0], 60.0533, rtol=0.001)
    assert_allclose(result.bse.iloc[0], 0.642, rtol=0.001)
    assert_allclose(result.cov_re.iloc[0, 0], 1.199, rtol=0.001)
    assert_allclose(result.scale, 0.67799, rtol=0.001)
    assert_allclose(result.llf, -123.997, rtol=0.1)
    assert_allclose(result.aic, 255.9944, rtol=0.001)
    assert_allclose(result.bic, 264.3718, rtol=0.001)