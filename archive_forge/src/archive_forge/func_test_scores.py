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
def test_scores(self):
    model, Z = (self.model, self.Z)
    for link in (links.Identity(), links.Log()):
        mod2 = BetaModel.from_formula(model, methylation, exog_precision=Z, link_precision=link)
        rslt_m = mod2.fit()
        analytical = rslt_m.model.score(rslt_m.params * 1.01)
        numerical = rslt_m.model._score_check(rslt_m.params * 1.01)
        assert_allclose(analytical, numerical, rtol=1e-06, atol=1e-06)
        assert_allclose(link.inverse(analytical[3:]), link.inverse(numerical[3:]), rtol=5e-07, atol=5e-06)