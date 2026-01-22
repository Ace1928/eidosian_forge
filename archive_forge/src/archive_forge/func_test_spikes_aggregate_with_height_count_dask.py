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
def test_spikes_aggregate_with_height_count_dask(self):
    spikes = Spikes([(1, 0.2), (2, 0.8), (3, 0.4)], vdims='y', datatype=['dask'])
    agg = rasterize(spikes, width=5, height=5, y_range=(0, 1), dynamic=False)
    xs = [1.2, 1.6, 2.0, 2.4, 2.8]
    ys = [0.1, 0.3, 0.5, 0.7, 0.9]
    arr = np.array([[1, 0, 1, 0, 1], [1, 0, 1, 0, 1], [0, 0, 1, 0, 1], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0]])
    expected = Image((xs, ys, arr), vdims=Dimension('Count', nodata=0))
    self.assertEqual(agg, expected)