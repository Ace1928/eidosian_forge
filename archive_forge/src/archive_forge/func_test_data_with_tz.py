import numpy as np
import pandas as pd
from holoviews.core.data import Dataset
from holoviews.core.data.interface import DataError
from holoviews.core.dimension import Dimension
from holoviews.core.spaces import HoloMap
from holoviews.element import Distribution, Points, Scatter
from .base import HeterogeneousColumnTests, InterfaceTests
def test_data_with_tz(self):
    dates = pd.date_range('2018-01-01', periods=3, freq='h')
    dates_tz = dates.tz_localize('UTC')
    df = pd.DataFrame({'dates': dates_tz})
    data = Dataset(df).dimension_values('dates')
    np.testing.assert_equal(dates, data)