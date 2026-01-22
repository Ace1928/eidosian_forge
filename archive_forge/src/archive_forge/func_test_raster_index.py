import numpy as np
from holoviews.element import HSV, RGB, Curve, Image, QuadMesh, Raster
from holoviews.element.comparison import ComparisonTestCase
def test_raster_index(self):
    raster = Raster(self.array1)
    self.assertEqual(raster[0, 1], 3)