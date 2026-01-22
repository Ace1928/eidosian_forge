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
def test_xarray_dataset_irregular_shape(self):
    ds = Dataset(self.get_multi_dim_irregular_dataset())
    shape = ds.interface.shape(ds, gridded=True)
    self.assertEqual(shape, (np.nan, np.nan, 3, 4))