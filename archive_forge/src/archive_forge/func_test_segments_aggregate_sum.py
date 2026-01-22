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
def test_segments_aggregate_sum(self, instance=False):
    segments = Segments([(0, 1, 4, 1, 2), (1, 0, 1, 4, 4)], vdims=['value'])
    if instance:
        agg = rasterize.instance(width=10, height=10, dynamic=False, aggregator='sum')(segments, width=4, height=4)
    else:
        agg = rasterize(segments, width=4, height=4, dynamic=False, aggregator='sum')
    xs = [0.5, 1.5, 2.5, 3.5]
    ys = [0.5, 1.5, 2.5, 3.5]
    na = np.nan
    arr = np.array([[na, 4, na, na], [2, 6, 2, 2], [na, 4, na, na], [na, 4, na, na]])
    expected = Image((xs, ys, arr), vdims='value')
    self.assertEqual(agg, expected)