import pytest
from holoviews.core.element import AttrTree
from holoviews.element.comparison import ComparisonTestCase
def test_uppercase_setitem(self):
    self.tree['C'] = 1
    self.assertEqual(self.tree.C, 1)