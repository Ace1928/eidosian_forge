import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pytest
from statsmodels.tools.tools import add_constant
from statsmodels.base._prediction_inference import PredictionResultsMonotonic
from statsmodels.discrete.discrete_model import (
from statsmodels.discrete.count_model import (
from statsmodels.sandbox.regression.tests.test_gmm_poisson import DATA
from .results import results_predict as resp
def test_predict_linear(self):
    res1 = self.res1
    ex = np.asarray(exog[:5])
    pred = res1.get_prediction(ex, which='linear', **self.pred_kwds_mean)
    k_extra = len(res1.params) - ex.shape[1]
    if k_extra > 0:
        ex = np.column_stack((ex, np.zeros((ex.shape[0], k_extra))))
    tt = res1.t_test(ex)
    cip = pred.conf_int()
    cit = tt.conf_int()
    assert_allclose(cip, cit, rtol=1e-12)