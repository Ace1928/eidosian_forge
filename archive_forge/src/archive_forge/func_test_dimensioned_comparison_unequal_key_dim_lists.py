from holoviews.core import Dimension, Dimensioned
from holoviews.element.comparison import ComparisonTestCase
def test_dimensioned_comparison_unequal_key_dim_lists(self):
    try:
        self.assertEqual(self.dimensioned1, self.dimensioned5)
    except AssertionError as e:
        self.assertEqual(str(e), 'Key dimension list mismatched')