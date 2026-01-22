import io
import os
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pandas as pd
import patsy
from statsmodels.api import families
from statsmodels.tools.sm_exceptions import (
from statsmodels.othermod.betareg import BetaModel
from .results import results_betareg as resultsb
def test_results_other(self):
    rslt = self.meth_fit
    distr = rslt.get_distribution()
    mean, var = distr.stats()
    assert_allclose(rslt.fittedvalues, mean, rtol=1e-13)
    assert_allclose(rslt.model._predict_var(rslt.params), var, rtol=1e-13)
    resid = rslt.model.endog - mean
    assert_allclose(rslt.resid, resid, rtol=1e-12)
    assert_allclose(rslt.resid_pearson, resid / np.sqrt(var), rtol=1e-12)