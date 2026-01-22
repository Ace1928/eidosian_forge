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
def test_line_rasterize(self):
    path = Path([[(0, 0), (1, 1), (2, 0)], [(0, 0), (0, 1)]], datatype=['spatialpandas'])
    agg = rasterize(path, width=4, height=4, dynamic=False)
    xs = [0.25, 0.75, 1.25, 1.75]
    ys = [0.125, 0.375, 0.625, 0.875]
    arr = np.array([[2, 0, 0, 1], [1, 1, 0, 1], [1, 1, 1, 0], [1, 0, 1, 0]])
    expected = Image((xs, ys, arr), vdims=Dimension('Count', nodata=0))
    self.assertEqual(agg, expected)