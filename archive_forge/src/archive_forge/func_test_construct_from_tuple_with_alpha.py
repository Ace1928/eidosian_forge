import numpy as np
from holoviews.element import HSV, RGB, Curve, Image, QuadMesh, Raster
from holoviews.element.comparison import ComparisonTestCase
def test_construct_from_tuple_with_alpha(self):
    rgb = RGB(([0, 1, 2], [0, 1, 2], self.rgb_array))
    self.assertEqual(len(rgb.vdims), 4)