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
@spatialpandas_skip
def test_polygon_rasterize_mean_agg(self):
    poly = Polygons([{'x': [0, 1, 2], 'y': [0, 1, 0], 'z': 2.4}, {'x': [0, 0, 1], 'y': [0, 1, 1], 'z': 3.6}], vdims='z')
    agg = rasterize(poly, width=4, height=4, dynamic=False, aggregator='mean')
    xs = [0.25, 0.75, 1.25, 1.75]
    ys = [0.125, 0.375, 0.625, 0.875]
    arr = np.array([[2.4, 2.4, 2.4, 2.4], [3.6, 2.4, 2.4, np.nan], [3.6, 2.4, 2.4, np.nan], [3.6, 3.6, np.nan, np.nan]])
    expected = Image((xs, ys, arr), vdims='z')
    self.assertEqual(agg, expected)