import os
import warnings
from statsmodels.compat.platform import PLATFORM_WIN
import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.statespace import sarimax, tools
from .results import results_sarimax
from statsmodels.tools import add_constant
from statsmodels.tools.tools import Bunch
from numpy.testing import (
def test_transform_untransform(self):
    model = self.model
    stat, inv = (model.enforce_stationarity, model.enforce_invertibility)
    true_constrained = self.true_params
    model.update(self.true_params)
    par = model.polynomial_ar
    psar = model.polynomial_seasonal_ar
    contracted_psar = psar[psar.nonzero()]
    model.enforce_stationarity = (model.k_ar == 0 or tools.is_invertible(np.r_[1, -par[1:]])) and (len(contracted_psar) <= 1 or tools.is_invertible(np.r_[1, -contracted_psar[1:]]))
    pma = model.polynomial_ma
    psma = model.polynomial_seasonal_ma
    contracted_psma = psma[psma.nonzero()]
    model.enforce_invertibility = (model.k_ma == 0 or tools.is_invertible(np.r_[1, pma[1:]])) and (len(contracted_psma) <= 1 or tools.is_invertible(np.r_[1, contracted_psma[1:]]))
    unconstrained = model.untransform_params(true_constrained)
    constrained = model.transform_params(unconstrained)
    assert_almost_equal(constrained, true_constrained, 4)
    model.enforce_stationarity = stat
    model.enforce_invertibility = inv