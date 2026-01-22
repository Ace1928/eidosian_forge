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
def test_rasterize_ndoverlay(self):
    ds = Dataset([(0.2, 0.3, 0), (0.4, 0.7, 1), (0, 0.99, 2)], kdims=['x', 'y', 'z'])
    ndoverlay = ds.to(Points, ['x', 'y'], [], 'z').overlay()
    expected = Image(([0.25, 0.75], [0.25, 0.75], [[1, 0], [2, 0]]), vdims=[Dimension('Count', nodata=0)])
    img = rasterize(ndoverlay, dynamic=False, x_range=(0, 1), y_range=(0, 1), width=2, height=2)
    self.assertEqual(img, expected)