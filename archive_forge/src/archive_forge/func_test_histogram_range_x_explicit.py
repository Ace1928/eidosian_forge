from holoviews import Dimension, ErrorBars, Histogram
from holoviews.element.comparison import ComparisonTestCase
def test_histogram_range_x_explicit(self):
    r = Histogram(([0, 1, 2, 3], [1, 2, 3]), kdims=[Dimension('x', range=(-1, 4.0))]).range(0)
    self.assertEqual(r, (-1.0, 4.0))