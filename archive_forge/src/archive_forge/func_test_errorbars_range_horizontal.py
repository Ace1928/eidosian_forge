from holoviews import Dimension, ErrorBars, Histogram
from holoviews.element.comparison import ComparisonTestCase
def test_errorbars_range_horizontal(self):
    r = ErrorBars(([1, 2, 3], [2, 3, 4], [0.5, 0.5, 0.5]), horizontal=True).range(0)
    self.assertEqual(r, (0.5, 3.5))