from unittest import SkipTest
from holoviews.element.comparison import ComparisonTestCase
from holoviews.core.options import Cycle, Options
def test_style_opts_advanced(self):
    line = "Layout (string='foo' test=3, b=True color=Cycle(values=[1,2]))"
    expected = {'Layout': {'style': Options(string='foo', test=3, b=True, color=Cycle(values=[1, 2]))}}
    self.assertEqual(OptsSpec.parse(line), expected)