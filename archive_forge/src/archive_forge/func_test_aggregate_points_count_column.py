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
def test_aggregate_points_count_column(self):
    points = Points([(0.2, 0.3, np.nan), (0.4, 0.7, 22), (0, 0.99, np.nan)], vdims='z')
    img = aggregate(points, dynamic=False, x_range=(0, 1), y_range=(0, 1), width=2, height=2, aggregator=ds.count('z'))
    expected = Image(([0.25, 0.75], [0.25, 0.75], [[0, 0], [1, 0]]), vdims=[Dimension('z Count', nodata=0)])
    self.assertEqual(img, expected)