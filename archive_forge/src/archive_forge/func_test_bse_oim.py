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
def test_bse_oim(self):
    bse = self.result._cov_params_oim().diagonal() ** 0.5
    assert_allclose(bse[0], self.true['se_ar_oim'], atol=0.1)
    assert_allclose(bse[1], self.true['se_ma_oim'], atol=0.1)