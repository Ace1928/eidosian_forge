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
def test_regrid_disabled_upsampling(self):
    img = Image(([0.5, 1.5], [0.5, 1.5], [[0, 1], [2, 3]]))
    regridded = regrid(img, width=3, height=3, dynamic=False, upsample=False)
    self.assertEqual(regridded, img)