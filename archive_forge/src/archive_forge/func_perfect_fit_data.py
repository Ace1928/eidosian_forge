import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import pytest
from scipy import stats
import statsmodels.api as sm
from statsmodels.robust import norms
from statsmodels.robust.robust_linear_model import RLM
from statsmodels.robust.scale import HuberScale
@pytest.fixture(scope='module')
def perfect_fit_data(request):
    from statsmodels.tools.tools import Bunch
    rs = np.random.RandomState(1249328932)
    exog = rs.standard_normal((1000, 1))
    endog = exog + exog ** 2
    exog = sm.add_constant(np.c_[exog, exog ** 2])
    return Bunch(endog=endog, exog=exog, const=3.2 * np.ones_like(endog))