import os
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
def test_autocorrelation_plot(self):
    from pandas.plotting import autocorrelation_plot
    ser = Series(np.arange(10, dtype=np.float64), index=date_range('2020-01-01', periods=10), name='ts')
    with tm.assert_produces_warning(None):
        _check_plot_works(autocorrelation_plot, series=ser)
        _check_plot_works(autocorrelation_plot, series=ser.values)
        ax = autocorrelation_plot(ser, label='Test')
    _check_legend_labels(ax, labels=['Test'])