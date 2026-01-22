from unittest import SkipTest
from holoviews.element.comparison import ComparisonTestCase
from holoviews.core.options import Cycle, Options
def test_combined_multiple_paths_merge_precedence(self):
    line = "Image (s=0, c='b') Image (s=3)"
    expected = {'Image': {'style': Options(c='b', s=3)}}
    self.assertEqual(OptsSpec.parse(line), expected)