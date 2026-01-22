import os
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
@pytest.mark.parametrize('kwargs', [{}, {'lag': 5}])
def test_lag_plot(self, kwargs):
    from pandas.plotting import lag_plot
    ser = Series(np.arange(10, dtype=np.float64), index=date_range('2020-01-01', periods=10), name='ts')
    _check_plot_works(lag_plot, series=ser, **kwargs)