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
def test_t_test(self):
    import statsmodels.tools._testing as smt
    smt.check_ttest_tvalues(self.result_b)
    smt.check_ftest_pvalues(self.result_b)