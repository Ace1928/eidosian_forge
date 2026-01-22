from statsmodels.compat.pandas import QUARTER_END, assert_index_equal
from statsmodels.compat.python import lrange
from io import BytesIO, StringIO
import os
import sys
import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal, assert_equal
import pandas as pd
import pytest
from statsmodels.datasets import macrodata
import statsmodels.tools.data as data_util
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.tsa.base.datetools import dates_from_str
import statsmodels.tsa.vector_ar.util as util
from statsmodels.tsa.vector_ar.var_model import VAR, var_acf
@pytest.mark.matplotlib
def test_plot_figsizes(self):
    assert_equal(self.irf.plot().get_size_inches(), (10, 10))
    assert_equal(self.irf.plot(figsize=(14, 10)).get_size_inches(), (14, 10))
    assert_equal(self.irf.plot_cum_effects().get_size_inches(), (10, 10))
    assert_equal(self.irf.plot_cum_effects(figsize=(14, 10)).get_size_inches(), (14, 10))