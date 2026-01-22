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
def test_oim(self):
    res1 = self.res1.model.fit()
    res2 = self.res2
    k_mean = 4
    p, se, zv, pv = res2.table_mean_oim.T
    assert_allclose(res1.params[:k_mean], p, rtol=1e-06)
    assert_allclose(res1.bse[:k_mean], se, rtol=1e-05)
    assert_allclose(res1.tvalues[:k_mean], zv, rtol=1e-05)
    assert_allclose(res1.pvalues[:k_mean], pv, atol=1e-06, rtol=1e-05)
    p, se, zv, pv = res2.table_precision_oim.T
    assert_allclose(res1.params[k_mean:], p, rtol=1e-06)
    assert_allclose(res1.bse[k_mean:], se, rtol=0.001)
    assert_allclose(res1.tvalues[k_mean:], zv, rtol=0.001)
    assert_allclose(res1.pvalues[k_mean:], pv, rtol=0.01)