import os
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
from scipy.linalg import cho_solve_banded
from statsmodels import datasets
from statsmodels.tsa.statespace import (sarimax, structural, dynamic_factor,
def test_posterior_mean(self):
    actual = np.array(self._sim_cfa.posterior_mean, copy=True)
    assert_allclose(actual, self.res.smoothed_state, atol=self.mean_atol)
    assert_allclose(self.sim_cfa.posterior_mean, self.res.smoothed_state, atol=self.mean_atol)