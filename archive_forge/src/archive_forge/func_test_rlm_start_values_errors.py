import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import pytest
from scipy import stats
import statsmodels.api as sm
from statsmodels.robust import norms
from statsmodels.robust.robust_linear_model import RLM
from statsmodels.robust.scale import HuberScale
def test_rlm_start_values_errors():
    data = sm.datasets.stackloss.load_pandas()
    exog = sm.add_constant(data.exog, prepend=False)
    model = RLM(data.endog, exog, M=norms.HuberT())
    start_params = [0.7156402, 1.29528612, -0.15212252]
    with pytest.raises(ValueError):
        model.fit(start_params=start_params)
    start_params = np.array([start_params, start_params]).T
    with pytest.raises(ValueError):
        model.fit(start_params=start_params)