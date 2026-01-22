from unittest import skipIf
import pandas as pd
import numpy as np
from holoviews import Curve, Scatter
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation.timeseries import resample, rolling, rolling_outlier_std
def test_roll_ints(self):
    rolled = rolling(self.int_curve, rolling_window=2)
    rolled_vals = [np.nan, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
    self.assertEqual(rolled, Curve(rolled_vals))