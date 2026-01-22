from unittest import SkipTest
from holoviews.element.comparison import ComparisonTestCase
from holoviews.core.options import Cycle, Options
def test_plot_opts_with_space(self):
    """Space in the tuple, see issue #77"""
    line = "Layout [fig_inches=(3, 3) title='foo bar']"
    expected = {'Layout': {'plot': Options(title='foo bar', fig_inches=(3, 3))}}
    self.assertEqual(OptsSpec.parse(line), expected)