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
def test_segments_aggregate_count(self):
    segments = Segments([(0, 1, 4, 1), (1, 0, 1, 4)])
    agg = rasterize(segments, width=4, height=4, dynamic=False)
    xs = [0.5, 1.5, 2.5, 3.5]
    ys = [0.5, 1.5, 2.5, 3.5]
    arr = np.array([[0, 1, 0, 0], [1, 2, 1, 1], [0, 1, 0, 0], [0, 1, 0, 0]])
    expected = Image((xs, ys, arr), vdims=Dimension('Count', nodata=0))
    self.assertEqual(agg, expected)