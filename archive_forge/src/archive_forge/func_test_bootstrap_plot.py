import os
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
def test_bootstrap_plot(self):
    from pandas.plotting import bootstrap_plot
    ser = Series(np.arange(10, dtype=np.float64), index=date_range('2020-01-01', periods=10), name='ts')
    _check_plot_works(bootstrap_plot, series=ser, size=10)