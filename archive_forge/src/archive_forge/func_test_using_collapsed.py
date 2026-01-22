import numpy as np
import pandas as pd
import pytest
import os
from statsmodels import datasets
from statsmodels.tsa.statespace import dynamic_factor
from statsmodels.tsa.statespace.mlemodel import MLEModel
from statsmodels.tsa.statespace.kalman_filter import (
from statsmodels.tsa.statespace.kalman_smoother import (
from statsmodels.tsa.statespace.tests.results import results_kalman_filter
from numpy.testing import assert_equal, assert_allclose
def test_using_collapsed(self):
    assert not self.results_a.filter_collapsed
    assert self.results_b.filter_collapsed
    assert self.results_a.collapsed_forecasts is None
    assert self.results_b.collapsed_forecasts is not None
    assert_equal(self.results_a.forecasts.shape[0], 3)
    assert_equal(self.results_b.collapsed_forecasts.shape[0], 2)