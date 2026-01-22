from statsmodels.compat.pytest import pytest_warns
from statsmodels.compat.pandas import assert_index_equal, assert_series_equal
from statsmodels.compat.platform import (
from statsmodels.compat.scipy import SCIPY_GT_14
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
import statsmodels.api as sm
from statsmodels.formula.api import glm, ols
import statsmodels.tools._testing as smt
from statsmodels.tools.sm_exceptions import HessianInversionWarning
def test_noformula(self):
    endog = self.res.model.endog
    exog = self.res.model.data.orig_exog
    exog = pd.DataFrame(exog)
    res = sm.OLS(endog, exog).fit()
    wa = res.wald_test_terms(skip_single=False, combine_terms=['Duration', 'Weight'], scalar=True)
    eye = np.eye(len(res.params))
    c_single = [row for row in eye]
    c_weight = eye[2:6]
    c_duration = eye[[1, 4, 5]]
    compare_waldres(res, wa, c_single + [c_duration, c_weight])
    df_constraints = [1] * len(c_single) + [3, 4]
    assert_equal(wa.df_constraints, df_constraints)