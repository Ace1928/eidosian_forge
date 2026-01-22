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
def test_rectangles_aggregate_count_cat(self):
    rects = Rectangles([(0, 0, 1, 2, 'A'), (1, 1, 3, 2, 'B')], vdims=['cat'])
    agg = rasterize(rects, width=4, height=4, aggregator=ds.count_cat('cat'), dynamic=False)
    xs = [0.375, 1.125, 1.875, 2.625]
    ys = [0.25, 0.75, 1.25, 1.75]
    arr1 = np.array([[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0]])
    arr2 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 1, 1], [0, 0, 0, 0]])
    expected1 = Image((xs, ys, arr1), vdims=Dimension('cat Count', nodata=0))
    expected2 = Image((xs, ys, arr2), vdims=Dimension('cat Count', nodata=0))
    expected = NdOverlay({'A': expected1, 'B': expected2}, kdims=['cat'])
    self.assertEqual(agg, expected)