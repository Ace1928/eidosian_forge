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
def test_dynamic_forecast(self):
    end = len(self.true['data']['consump']) + 15 - 1
    dynamic = len(self.true['data']['consump']) - 1
    exog = add_constant(self.true['forecast_data']['m2'])
    assert_almost_equal(self.result.predict(end=end, dynamic=dynamic, exog=exog), self.true['dynamic_forecast'], 3)