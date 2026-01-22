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
def test_rasterize_regrid_and_spikes_overlay(self):
    img = Image(([0.5, 1.5], [0.5, 1.5], [[0, 1], [2, 3]]))
    spikes = Spikes([(0.5, 0.2), (1.5, 0.8)], vdims='y')
    expected_regrid = Image(([0.25, 0.75, 1.25, 1.75], [0.25, 0.75, 1.25, 1.75], [[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 3, 3], [2, 2, 3, 3]]))
    spikes_arr = np.array([[0, 1, 0, 1], [0, 1, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])
    expected_spikes = Image(([0.25, 0.75, 1.25, 1.75], [0.25, 0.75, 1.25, 1.75], spikes_arr), vdims=Dimension('Count', nodata=0))
    overlay = img * spikes
    agg = rasterize(overlay, width=4, height=4, x_range=(0, 2), y_range=(0, 2), spike_length=0.5, upsample=True, dynamic=False)
    self.assertEqual(agg.Image.I, expected_regrid)
    self.assertEqual(agg.Spikes.I, expected_spikes)