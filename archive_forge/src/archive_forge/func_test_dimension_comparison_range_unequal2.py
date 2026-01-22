from holoviews.core import Dimension, Dimensioned
from holoviews.element.comparison import ComparisonTestCase
def test_dimension_comparison_range_unequal2(self):
    try:
        self.assertEqual(self.dimension5, self.dimension6)
    except AssertionError as e:
        self.assertEqual(str(e), "Dimension parameter 'range' mismatched: (None, None) != (0, 1)")