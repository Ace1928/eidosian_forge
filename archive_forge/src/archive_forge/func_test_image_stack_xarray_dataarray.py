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
def test_image_stack_xarray_dataarray(self):
    x = np.arange(0, 3)
    y = np.arange(5, 8)
    a = np.array([[np.nan, np.nan, 1], [np.nan] * 3, [np.nan] * 3])
    b = np.array([[np.nan] * 3, [1, 1, np.nan], [np.nan] * 3])
    c = np.array([[np.nan] * 3, [np.nan] * 3, [1, 1, 1]])
    ds = xr.Dataset({'a': (['x', 'y'], a), 'b': (['x', 'y'], b), 'c': (['x', 'y'], c)}, coords={'x': x, 'y': y}).to_array('level')
    img_stack = ImageStack(ds, vdims=['level'])
    assert img_stack.interface is XArrayInterface
    assert img_stack.kdims == [Dimension('x'), Dimension('y')]
    assert img_stack.vdims == [Dimension('a'), Dimension('b'), Dimension('c')]