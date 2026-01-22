from holoviews import Dimension, ErrorBars, Histogram
from holoviews.element.comparison import ComparisonTestCase
def test_histogram_range_y_explicit_lower(self):
    r = Histogram(([0, 1, 2, 3], [1, 2, 3]), vdims=[Dimension('y', range=(0.0, None))]).range(1)
    self.assertEqual(r, (0.0, 3.0))