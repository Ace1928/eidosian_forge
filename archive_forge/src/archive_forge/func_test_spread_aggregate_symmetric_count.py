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
def test_spread_aggregate_symmetric_count(self):
    spread = Spread([(0, 1, 0.8), (1, 2, 0.3), (2, 3, 0.8)])
    agg = rasterize(spread, width=4, height=4, dynamic=False)
    xs = [0.25, 0.75, 1.25, 1.75]
    ys = [0.65, 1.55, 2.45, 3.35]
    arr = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 1, 0], [0, 0, 0, 1]])
    expected = Image((xs, ys, arr), vdims=Dimension('Count', nodata=0))
    self.assertEqual(agg, expected)