import numpy as np
from holoviews.core import BoundingBox
from holoviews.element.comparison import ComparisonTestCase
def test_bounds_equal(self):
    self.assertEqual(BoundingBox(radius=0.5), BoundingBox(radius=0.5))