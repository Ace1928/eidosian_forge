from holoviews.core import Dimension, Dimensioned
from holoviews.element.comparison import ComparisonTestCase
def test_dimension_comparison_value_format_unequal(self):
    self.assertEqual(self.dimension12, self.dimension13)
    self.assertNotEqual(str(self.dimension12.value_format), str(self.dimension13.value_format))