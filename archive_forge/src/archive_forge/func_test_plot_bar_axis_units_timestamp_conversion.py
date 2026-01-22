import os
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
def test_plot_bar_axis_units_timestamp_conversion(self):
    df = DataFrame([1.0], index=[Timestamp('2022-02-22 22:22:22')])
    _check_plot_works(df.plot)
    s = Series({'A': 1.0})
    _check_plot_works(s.plot.bar)