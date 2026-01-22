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
def test_rasterize_image(self):
    img = Image((range(10), range(5), np.arange(10) * np.arange(5)[np.newaxis].T))
    regridded = regrid(img, width=2, height=2, dynamic=False)
    expected = Image(([2.0, 7.0], [0.75, 3.25], [[1, 5], [6, 22]]))
    self.assertEqual(regridded, expected)