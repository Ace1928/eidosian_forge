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
def test_regrid_disabled_expand(self):
    img = Image(([0.5, 1.5], [0.5, 1.5], [[0.0, 1.0], [2.0, 3.0]]))
    regridded = regrid(img, width=2, height=2, x_range=(-2, 4), y_range=(-2, 4), expand=False, dynamic=False)
    self.assertEqual(regridded, img)