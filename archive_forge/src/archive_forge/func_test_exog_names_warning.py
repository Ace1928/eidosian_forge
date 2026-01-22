import pytest
import numpy as np
from numpy.testing import assert_almost_equal
from scipy import stats
import statsmodels.api as sm
from statsmodels.miscmodels.count import PoissonGMLE, PoissonOffsetGMLE, \
from statsmodels.discrete.discrete_model import Poisson
from statsmodels.tools.sm_exceptions import ValueWarning
def test_exog_names_warning(self):
    mod = self.res.model
    mod1 = PoissonOffsetGMLE(mod.endog, mod.exog, offset=mod.offset)
    from numpy.testing import assert_warns
    mod1.data.xnames = mod1.data.xnames * 2
    assert_warns(ValueWarning, mod1.fit, disp=0)