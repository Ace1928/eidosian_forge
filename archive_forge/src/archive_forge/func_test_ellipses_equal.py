from holoviews import Bounds, Box, Contours, Ellipse, Path
from holoviews.element.comparison import ComparisonTestCase
def test_ellipses_equal(self):
    self.assertEqual(self.ellipse1, self.ellipse1)