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
def test_inspect_points_or_polygons(self):
    if spatialpandas is None:
        raise SkipTest('Polygon inspect tests require spatialpandas')
    polys = inspect(self.polysrgb, max_indicators=3, dynamic=False, pixels=1, x=6, y=5)
    self.assertEqual(polys, Polygons([{'x': [6, 3, 7], 'y': [7, 2, 5], 'z': 2}], vdims='z'))
    points = inspect(self.pntsimg, max_indicators=3, dynamic=False, pixels=1, x=-0.1, y=-0.1)
    self.assertEqual(points.dimension_values('x'), np.array([]))
    self.assertEqual(points.dimension_values('y'), np.array([]))