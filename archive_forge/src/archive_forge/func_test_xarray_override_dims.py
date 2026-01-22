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
def test_xarray_override_dims(self):
    xs = [0.1, 0.2, 0.3]
    ys = [0, 1]
    zs = np.array([[0, 1], [2, 3], [4, 5]])
    da = xr.DataArray(zs, coords=[('x_dim', xs), ('y_dim', ys)], name='data_name', dims=['y_dim', 'x_dim'])
    da.attrs['long_name'] = 'data long name'
    da.attrs['units'] = 'array_unit'
    da.x_dim.attrs['units'] = 'x_unit'
    da.y_dim.attrs['long_name'] = 'y axis long name'
    ds = Dataset(da, kdims=['x_dim', 'y_dim'], vdims=['z_dim'])
    x_dim = Dimension('x_dim')
    y_dim = Dimension('y_dim')
    z_dim = Dimension('z_dim')
    self.assertEqual(ds.kdims[0], x_dim)
    self.assertEqual(ds.kdims[1], y_dim)
    self.assertEqual(ds.vdims[0], z_dim)
    ds_from_ds = Dataset(da.to_dataset(), kdims=['x_dim', 'y_dim'], vdims=['data_name'])
    self.assertEqual(ds_from_ds.kdims[0], x_dim)
    self.assertEqual(ds_from_ds.kdims[1], y_dim)
    data_dim = Dimension('data_name')
    self.assertEqual(ds_from_ds.vdims[0], data_dim)