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
def test_dietox_slopes(self):
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    rdir = os.path.join(cur_dir, 'results')
    fname = os.path.join(rdir, 'dietox.csv')
    data = pd.read_csv(fname)
    model = MixedLM.from_formula('Weight ~ Time', groups='Pig', re_formula='1 + Time', data=data)
    result = model.fit(method='cg')
    assert_allclose(result.fe_params, np.r_[15.73865, 6.939014], rtol=1e-05)
    assert_allclose(result.bse[0:2], np.r_[0.5501253, 0.0798254], rtol=0.001)
    assert_allclose(result.scale, 6.03745, rtol=0.001)
    assert_allclose(result.cov_re.values.ravel(), np.r_[19.4934552, 0.2938323, 0.2938323, 0.416062], rtol=0.1)
    assert_allclose(model.loglike(result.params_object), -2217.047, rtol=1e-05)
    data = pd.read_csv(fname)
    model = MixedLM.from_formula('Weight ~ Time', groups='Pig', re_formula='1 + Time', data=data)
    result = model.fit(method='cg', reml=False)
    assert_allclose(result.fe_params, np.r_[15.73863, 6.93902], rtol=1e-05)
    assert_allclose(result.bse[0:2], np.r_[0.54629282, 0.07926954], rtol=0.001)
    assert_allclose(result.scale, 6.037441, rtol=0.001)
    assert_allclose(result.cov_re.values.ravel(), np.r_[19.190922, 0.293568, 0.293568, 0.409695], rtol=0.01)
    assert_allclose(model.loglike(result.params_object), -2215.753, rtol=1e-05)