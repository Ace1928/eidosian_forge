from holoviews import Bounds, Box, Contours, Ellipse, Path
from holoviews.element.comparison import ComparisonTestCase
def test_boxs_equal(self):
    self.assertEqual(self.box1, self.box1)