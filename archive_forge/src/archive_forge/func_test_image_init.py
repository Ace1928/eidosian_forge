import numpy as np
import holoviews as hv
from holoviews.element import Curve, Image
from ..utils import LoggingComparisonTestCase
def test_image_init(self):
    image = Image(self.array1)
    self.assertEqual(image.xdensity, 3)
    self.assertEqual(image.ydensity, 2)