from holoviews.core import Dimension, Element
from holoviews.element.comparison import ComparisonTestCase
def test_value_dimension_in_element(self):
    self.assertTrue(Dimension('C') in self.element)