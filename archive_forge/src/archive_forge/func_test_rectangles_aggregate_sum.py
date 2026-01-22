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
def test_rectangles_aggregate_sum(self):
    rects = Rectangles([(0, 0, 1, 2, 0.5), (1, 1, 3, 2, 1.5)], vdims=['value'])
    agg = rasterize(rects, width=4, height=4, aggregator='sum', dynamic=False)
    xs = [0.375, 1.125, 1.875, 2.625]
    ys = [0.25, 0.75, 1.25, 1.75]
    arr = np.array([[0.5, 0.5, nan, nan], [0.5, 0.5, nan, nan], [0.5, 2.0, 1.5, 1.5], [nan, nan, nan, nan]])
    expected = Image((xs, ys, arr), vdims='value')
    self.assertEqual(agg, expected)