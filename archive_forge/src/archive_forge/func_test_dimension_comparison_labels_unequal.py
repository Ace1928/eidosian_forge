from holoviews.core import Dimension, Dimensioned
from holoviews.element.comparison import ComparisonTestCase
def test_dimension_comparison_labels_unequal(self):
    try:
        self.assertEqual(self.dimension1, self.dimension11)
    except AssertionError as e:
        self.assertEqual(str(e), 'Dimension labels mismatched: dim1 != Test Dimension')