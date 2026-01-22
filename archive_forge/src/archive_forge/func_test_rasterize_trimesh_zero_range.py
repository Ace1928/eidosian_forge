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
def test_rasterize_trimesh_zero_range(self):
    trimesh = TriMesh((self.simplexes_vdim, self.vertices), vdims=['z'])
    img = rasterize(trimesh, x_range=(0, 0), height=2, dynamic=False)
    image = Image(([], [0.25, 0.75], np.zeros((2, 0))), bounds=(0, 0, 0, 1), xdensity=1)
    self.assertEqual(img, image)