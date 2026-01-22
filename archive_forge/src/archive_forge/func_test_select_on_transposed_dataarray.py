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
def test_select_on_transposed_dataarray(self):
    x = np.linspace(-3, 7, 53)
    y = np.linspace(-5, 8, 89)
    z = np.exp(-1 * (x ** 2 + y[:, np.newaxis] ** 2))
    array = xr.DataArray(z, coords=[y, x], dims=['x', 'y'])
    img = Image(array)[1:3]
    self.assertEqual(img['z'], Image(array.sel(x=slice(1, 3)))['z'])