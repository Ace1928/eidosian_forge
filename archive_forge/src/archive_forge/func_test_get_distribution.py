import itertools
import os
import numpy as np
from statsmodels.duration.hazard_regression import PHReg
from numpy.testing import (assert_allclose,
import pandas as pd
import pytest
from .results import survival_r_results
from .results import survival_enet_r_results
@pytest.mark.smoke
def test_get_distribution(self):
    np.random.seed(34234)
    n = 200
    exog = np.random.normal(size=(n, 2))
    lin_pred = exog.sum(1)
    elin_pred = np.exp(-lin_pred)
    time = -elin_pred * np.log(np.random.uniform(size=n))
    status = np.ones(n)
    status[0:20] = 0
    strata = np.kron(range(5), np.ones(n // 5))
    mod = PHReg(time, exog, status=status, strata=strata)
    rslt = mod.fit()
    dist = rslt.get_distribution()
    fitted_means = dist.mean()
    true_means = elin_pred
    fitted_var = dist.var()
    fitted_sd = dist.std()
    sample = dist.rvs()