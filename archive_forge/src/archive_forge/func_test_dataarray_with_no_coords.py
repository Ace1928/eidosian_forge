import datetime as dt
from unittest import SkipTest
import numpy as np
import pandas as pd
from holoviews.core.data import Dataset, XArrayInterface, concat
from holoviews.core.dimension import Dimension
from holoviews.core.spaces import HoloMap
from holoviews.element import HSV, RGB, Image, ImageStack, QuadMesh
from .test_gridinterface import BaseGridInterfaceTests
from .test_imageinterface import (
def test_dataarray_with_no_coords(self):
    expected_xs = list(range(2))
    expected_ys = list(range(3))
    zs = np.arange(6).reshape(2, 3)
    xrarr = xr.DataArray(zs, dims=('x', 'y'))
    img = Image(xrarr)
    self.assertTrue(all(img.data.x == expected_xs))
    self.assertTrue(all(img.data.y == expected_ys))
    img = Image(xrarr, kdims=['x', 'y'])
    self.assertTrue(all(img.data.x == expected_xs))
    self.assertTrue(all(img.data.y == expected_ys))