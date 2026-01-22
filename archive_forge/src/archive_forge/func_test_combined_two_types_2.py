from unittest import SkipTest
from holoviews.element.comparison import ComparisonTestCase
from holoviews.core.options import Cycle, Options
def test_combined_two_types_2(self):
    line = "Layout plot[fig_inches=(3, 3)] Image (string='foo') [foo='bar baz']"
    expected = {'Layout': {'plot': Options(fig_inches=(3, 3))}, 'Image': {'style': Options(string='foo'), 'plot': Options(foo='bar baz')}}
    self.assertEqual(OptsSpec.parse(line), expected)