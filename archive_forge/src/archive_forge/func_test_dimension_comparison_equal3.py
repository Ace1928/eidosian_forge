from holoviews.core import Dimension, Dimensioned
from holoviews.element.comparison import ComparisonTestCase
def test_dimension_comparison_equal3(self):
    self.assertEqual(self.dimension7, Dimension('dim1', cyclic=True, range=(0, 1), unit='ms'))