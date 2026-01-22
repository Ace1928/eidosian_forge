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
def init_column_data(self):
    import dask.array
    self.xs = np.array(range(11))
    self.xs_2 = self.xs ** 2
    self.y_ints = self.xs * 2
    dask_y = dask.array.from_array(np.array(self.y_ints), 2)
    self.dataset_hm = Dataset((self.xs, dask_y), kdims=['x'], vdims=['y'])
    self.dataset_hm_alias = Dataset((self.xs, dask_y), kdims=[('x', 'X')], vdims=[('y', 'Y')])