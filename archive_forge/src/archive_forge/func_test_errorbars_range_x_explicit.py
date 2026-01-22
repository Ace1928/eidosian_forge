from holoviews import Dimension, ErrorBars, Histogram
from holoviews.element.comparison import ComparisonTestCase
def test_errorbars_range_x_explicit(self):
    r = ErrorBars(([1, 2, 3], [2, 3, 4], [0.5, 0.5, 0.5]), kdims=[Dimension('x', range=(-1, 4.0))]).range(0)
    self.assertEqual(r, (-1.0, 4.0))