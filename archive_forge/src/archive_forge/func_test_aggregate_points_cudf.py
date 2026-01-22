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
@cudf_skip
def test_aggregate_points_cudf(self):
    points = Points([(0.2, 0.3), (0.4, 0.7), (0, 0.99)], datatype=['cuDF'])
    self.assertIsInstance(points.data, cudf.DataFrame)
    img = aggregate(points, dynamic=False, x_range=(0, 1), y_range=(0, 1), width=2, height=2)
    expected = Image(([0.25, 0.75], [0.25, 0.75], [[1, 0], [2, 0]]), vdims=[Dimension('Count', nodata=0)])
    self.assertIsInstance(img.data.Count.data, cupy.ndarray)
    self.assertEqual(img, expected)