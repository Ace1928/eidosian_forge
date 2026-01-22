from unittest import SkipTest
import numpy as np
from holoviews.core.data import Dataset, concat
from geoviews.data.iris import coord_to_dimension
from holoviews.core.spaces import HoloMap
from holoviews.element import Image
from holoviews.tests.core.data.test_imageinterface import BaseImageElementInterfaceTests
from holoviews.tests.core.data.test_gridinterface import BaseGridInterfaceTests
def test_initialize_cube_with_kdims(self):
    cube = Dataset(self.cube, kdims=['longitude', 'latitude'])
    self.assertEqual(cube.dimensions('key', True), ['longitude', 'latitude'])