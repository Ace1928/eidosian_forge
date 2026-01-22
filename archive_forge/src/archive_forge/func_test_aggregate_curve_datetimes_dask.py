import datetime as dt
from unittest import SkipTest, skipIf
import colorcet as cc
import numpy as np
import pandas as pd
import pytest
from numpy import nan
from packaging.version import Version
from holoviews import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import apply_when
from holoviews.streams import Tap
from holoviews.util import render
import logging
def test_aggregate_curve_datetimes_dask(self):
    df = pd.DataFrame(data=np.arange(1000), columns=['a'], index=pd.date_range('2019-01-01', freq='1min', periods=1000))
    ddf = dd.from_pandas(df, npartitions=4)
    curve = Curve(ddf, kdims=['index'], vdims=['a'])
    img = aggregate(curve, width=2, height=3, dynamic=False)
    bounds = (np.datetime64('2019-01-01T00:00:00.000000'), 0.0, np.datetime64('2019-01-01T16:39:00.000000'), 999.0)
    dates = [np.datetime64('2019-01-01T04:09:45.000000000'), np.datetime64('2019-01-01T12:29:15.000000000')]
    expected = Image((dates, [166.5, 499.5, 832.5], [[332, 0], [167, 166], [0, 334]]), kdims=['index', 'a'], vdims=Dimension('Count', nodata=0), datatype=['xarray'], bounds=bounds)
    self.assertEqual(img, expected)