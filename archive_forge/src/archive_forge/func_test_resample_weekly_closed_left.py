from unittest import skipIf
import pandas as pd
import numpy as np
from holoviews import Curve, Scatter
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation.timeseries import resample, rolling, rolling_outlier_std
def test_resample_weekly_closed_left(self):
    resampled = resample(self.date_curve, rule='W', closed='left')
    dates = list(map(pd.Timestamp, ['2016-01-03', '2016-01-10']))
    vals = [1.5, 5]
    self.assertEqual(resampled, Curve((dates, vals)))