import numpy as np
from holoviews.element import HSV, RGB, Curve, Image, QuadMesh, Raster
from holoviews.element.comparison import ComparisonTestCase
def test_construct_from_array_with_alpha(self):
    rgb = RGB(self.rgb_array)
    self.assertEqual(len(rgb.vdims), 4)