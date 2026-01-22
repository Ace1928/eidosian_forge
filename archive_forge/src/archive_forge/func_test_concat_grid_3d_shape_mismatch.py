from unittest import SkipTest
import numpy as np
from holoviews.core.data import Dataset, concat
from geoviews.data.iris import coord_to_dimension
from holoviews.core.spaces import HoloMap
from holoviews.element import Image
from holoviews.tests.core.data.test_imageinterface import BaseImageElementInterfaceTests
from holoviews.tests.core.data.test_gridinterface import BaseGridInterfaceTests
def test_concat_grid_3d_shape_mismatch(self):
    arr1 = np.random.rand(3, 2)
    arr2 = np.random.rand(2, 3)
    ds1 = Dataset(([0, 1], [1, 2, 3], arr1), ['x', 'y'], 'z')
    ds2 = Dataset(([0, 1, 2], [1, 2], arr2), ['x', 'y'], 'z')
    hmap = HoloMap({1: ds1, 2: ds2})
    with self.assertRaises(MergeError):
        concat(hmap)