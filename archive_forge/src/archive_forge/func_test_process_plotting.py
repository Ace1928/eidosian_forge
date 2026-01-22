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
def test_process_plotting(self, close_figures):
    res0 = self.res0
    k_ar = res0.k_ar
    fc20 = res0.forecast(res0.endog[-k_ar:], 20)
    irf = res0.irf()
    res0.plotsim()
    res0.plot_acorr()
    fig = res0.plot_forecast(20)
    fcp = fig.axes[0].get_children()[1].get_ydata()[-20:]
    assert_allclose(fc20[:, 0], fcp, rtol=1e-13)
    fcp = fig.axes[1].get_children()[1].get_ydata()[-20:]
    assert_allclose(fc20[:, 1], fcp, rtol=1e-13)
    fcp = fig.axes[2].get_children()[1].get_ydata()[-20:]
    assert_allclose(fc20[:, 2], fcp, rtol=1e-13)
    fig_asym = irf.plot()
    fig_mc = irf.plot(stderr_type='mc', repl=1000, seed=987128)
    for k in range(3):
        a = fig_asym.axes[1].get_children()[k].get_ydata()
        m = fig_mc.axes[1].get_children()[k].get_ydata()
        assert_allclose(a, m, atol=0.1, rtol=0.9)