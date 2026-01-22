from unittest import SkipTest
from holoviews.element.comparison import ComparisonTestCase
from holoviews.core.options import Cycle, Options
def test_combined_two_types_1(self):
    line = "Layout plot[fig_inches=(3,3) foo='bar baz'] Image (string='foo')"
    expected = {'Layout': {'plot': Options(foo='bar baz', fig_inches=(3, 3))}, 'Image': {'style': Options(string='foo')}}
    self.assertEqual(OptsSpec.parse(line), expected)