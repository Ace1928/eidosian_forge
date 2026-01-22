from holoviews.core import Dimension, Dimensioned
from holoviews.element.comparison import ComparisonTestCase
def test_dimension_comparison_types_unequal(self):
    try:
        self.assertEqual(self.dimension9, self.dimension10)
    except AssertionError as e:
        self.assertEqual(str(e), "Dimension parameter 'type' mismatched: <class 'int'> != <class 'float'>")