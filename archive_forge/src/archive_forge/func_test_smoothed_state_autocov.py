import os
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal, assert_equal
import pandas as pd
import pytest
from statsmodels import datasets
from statsmodels.tsa.statespace import mlemodel, sarimax, varmax
from statsmodels.tsa.statespace.tests.test_impulse_responses import TVSS
from statsmodels.tsa.statespace.kalman_filter import FILTER_UNIVARIATE
from statsmodels.tsa.statespace.kalman_smoother import (
def test_smoothed_state_autocov(self):
    assert_allclose(self.results.smoothed_state_autocov[:, :, 0:5], self.augmented_results.smoothed_state_cov[:3, 3:, 1:6], atol=0.0001)
    assert_allclose(self.results.smoothed_state_autocov[:, :, 5:-1], self.augmented_results.smoothed_state_cov[:3, 3:, 6:], atol=1e-07)