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
def test_from_empty_numpy(self):
    """
        Datashader sometimes pass an empty array to the interface
        """
    kdims = ['dim_0', 'dim_1']
    vdims = ['dim_2']
    ds = XArrayInterface.init(Image, np.array([]), kdims, vdims)
    assert isinstance(ds[0], xr.Dataset)
    assert ds[0][vdims[0]].size == 0
    assert ds[1]['kdims'] == kdims
    assert ds[1]['vdims'] == vdims