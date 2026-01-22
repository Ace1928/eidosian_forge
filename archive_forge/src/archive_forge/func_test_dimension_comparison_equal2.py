from holoviews.core import Dimension, Dimensioned
from holoviews.element.comparison import ComparisonTestCase
def test_dimension_comparison_equal2(self):
    self.assertEqual(self.dimension1, Dimension('dim1', range=(0, 1)))