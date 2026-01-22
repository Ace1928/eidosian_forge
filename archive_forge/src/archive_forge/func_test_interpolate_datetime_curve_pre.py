import datetime as dt
from unittest import skipIf
import numpy as np
import pandas as pd
import pytest
from holoviews import (
from holoviews.core.data.grid import GridInterface
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation.element import (
def test_interpolate_datetime_curve_pre(self):
    dates = np.array([dt.datetime(2017, 1, i) for i in range(1, 5)]).astype('M')
    values = [0, 1, 2, 3]
    interpolated = interpolate_curve(Curve((dates, values)), interpolation='steps-pre')
    dates_interp = np.array(['2017-01-01T00:00:00', '2017-01-01T00:00:00', '2017-01-02T00:00:00', '2017-01-02T00:00:00', '2017-01-03T00:00:00', '2017-01-03T00:00:00', '2017-01-04T00:00:00'], dtype='datetime64[ns]')
    curve = Curve((dates_interp, [0, 1, 1, 2, 2, 3, 3]))
    self.assertEqual(interpolated, curve)