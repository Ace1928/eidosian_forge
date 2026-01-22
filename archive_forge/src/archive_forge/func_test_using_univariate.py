import os
import numpy as np
from numpy.testing import assert_almost_equal, assert_allclose
import pandas as pd
import pytest
from statsmodels import datasets
from statsmodels.tsa.statespace.mlemodel import MLEModel
from statsmodels.tsa.statespace.tests.results import results_kalman_filter
from statsmodels.tsa.statespace.sarimax import SARIMAX
def test_using_univariate(self):
    assert not self.conventional_results.filter_univariate
    assert self.univariate_results.filter_univariate
    assert_allclose(self.conventional_results.forecasts_error_cov[1, 1, 0], 1000000.77)
    assert_allclose(self.univariate_results.forecasts_error_cov[1, 1, 0], 1000000.77)