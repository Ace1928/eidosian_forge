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
def test_xarray_irregular_dataset_values(self):
    ds = Dataset(self.get_multi_dim_irregular_dataset())
    values = ds.dimension_values('z', expanded=False)
    self.assertEqual(values, np.array([0, 1, 2, 3]))