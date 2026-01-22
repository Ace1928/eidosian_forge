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
def test_vcomp_1(self):
    np.random.seed(4279)
    exog = np.random.normal(size=(400, 1))
    exog_re = np.random.normal(size=(400, 2))
    groups = np.kron(np.arange(100), np.ones(4))
    slopes = np.random.normal(size=(100, 2))
    slopes[:, 1] *= 2
    slopes = np.kron(slopes, np.ones((4, 1))) * exog_re
    errors = slopes.sum(1) + np.random.normal(size=400)
    endog = exog.sum(1) + errors
    free = MixedLMParams(1, 2, 0)
    free.fe_params = np.ones(1)
    free.cov_re = np.eye(2)
    free.vcomp = np.zeros(0)
    model1 = MixedLM(endog, exog, groups, exog_re=exog_re)
    result1 = model1.fit(free=free)
    exog_vc = {'a': {}, 'b': {}}
    for k, group in enumerate(model1.group_labels):
        ix = model1.row_indices[group]
        exog_vc['a'][group] = exog_re[ix, 0:1]
        exog_vc['b'][group] = exog_re[ix, 1:2]
    with pytest.warns(UserWarning, match='Using deprecated variance'):
        model2 = MixedLM(endog, exog, groups, exog_vc=exog_vc)
    result2 = model2.fit()
    result2.summary()
    assert_allclose(result1.fe_params, result2.fe_params, atol=0.0001)
    assert_allclose(np.diag(result1.cov_re), result2.vcomp, atol=0.01, rtol=0.0001)
    assert_allclose(result1.bse[[0, 1, 3]], result2.bse, atol=0.01, rtol=0.01)