import os
import warnings
from statsmodels.compat.platform import PLATFORM_WIN
import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.statespace import sarimax, tools
from .results import results_sarimax
from statsmodels.tools import add_constant
from statsmodels.tools.tools import Bunch
from numpy.testing import (
def test_regression_coefficient(self):
    self.result = self.model.filter(self.true_params)
    assert_allclose(self.result.filter_results.filtered_state[3][-1], self.true_regression_coefficient, self.decimal)