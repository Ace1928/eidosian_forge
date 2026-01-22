from statsmodels.compat.pandas import (
import numpy as np
from numpy.testing import assert_equal, assert_, assert_raises
import pandas as pd
import pytest
from statsmodels.base import data as sm_data
from statsmodels.formula import handle_formula_data
from statsmodels.regression.linear_model import OLS
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families
from statsmodels.discrete.discrete_model import Logit
def test_hasconst(self):
    for x, result in zip(self.exogs, self.results):
        mod = self.mod(self.y, x)
        assert_equal(mod.k_constant, result[0])
        assert_equal(mod.data.k_constant, result[0])
        if result[1] is None:
            assert_(mod.data.const_idx is None)
        else:
            assert_equal(mod.data.const_idx, result[1])
        fit_kwds = getattr(self, 'fit_kwds', {})
        try:
            res = mod.fit(**fit_kwds)
        except np.linalg.LinAlgError:
            pass
        else:
            assert_equal(res.model.k_constant, result[0])
            assert_equal(res.model.data.k_constant, result[0])