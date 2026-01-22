import datetime as dt
from unittest import skipIf
import numpy as np
import pandas as pd
import pytest
from holoviews import (
from holoviews.core.data.grid import GridInterface
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation.element import (
def test_histogram_operation_datetime(self):
    dates = np.array([dt.datetime(2017, 1, i) for i in range(1, 5)])
    op_hist = histogram(Dataset(dates, 'Date'), num_bins=4, normed=True)
    hist_data = {'Date': np.array(['2017-01-01T00:00:00.000000', '2017-01-01T18:00:00.000000', '2017-01-02T12:00:00.000000', '2017-01-03T06:00:00.000000', '2017-01-04T00:00:00.000000'], dtype='datetime64[us]'), 'Date_frequency': np.array([3.85802469e-18, 3.85802469e-18, 3.85802469e-18, 3.85802469e-18])}
    hist = Histogram(hist_data, kdims='Date', vdims=('Date_frequency', 'Frequency'))
    self.assertEqual(op_hist, hist)