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
def init_grid_data(self):
    import dask.array
    self.grid_xs = [0, 1]
    self.grid_ys = [0.1, 0.2, 0.3]
    self.grid_zs = np.array([[0, 1], [2, 3], [4, 5]])
    dask_zs = dask.array.from_array(self.grid_zs, 2)
    self.dataset_grid = self.element((self.grid_xs, self.grid_ys, dask_zs), kdims=['x', 'y'], vdims=['z'])
    self.dataset_grid_alias = self.element((self.grid_xs, self.grid_ys, dask_zs), kdims=[('x', 'X'), ('y', 'Y')], vdims=[('z', 'Z')])
    self.dataset_grid_inv = self.element((self.grid_xs[::-1], self.grid_ys[::-1], dask_zs), kdims=['x', 'y'], vdims=['z'])