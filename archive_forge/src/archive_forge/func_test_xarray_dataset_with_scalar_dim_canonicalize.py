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
def test_xarray_dataset_with_scalar_dim_canonicalize(self):
    import dask.array
    xs = [0, 1]
    ys = [0.1, 0.2, 0.3]
    zs = dask.array.from_array(np.array([[[0, 1], [2, 3], [4, 5]]]), 2)
    xrarr = xr.DataArray(zs, coords={'x': xs, 'y': ys, 't': [1]}, dims=['t', 'y', 'x'])
    xrds = xr.Dataset({'v': xrarr})
    ds = Dataset(xrds, kdims=['x', 'y'], vdims=['v'], datatype=['xarray'])
    canonical = ds.dimension_values(2, flat=False)
    self.assertEqual(canonical.ndim, 2)
    expected = np.array([[0, 1], [2, 3], [4, 5]])
    self.assertEqual(canonical, expected)