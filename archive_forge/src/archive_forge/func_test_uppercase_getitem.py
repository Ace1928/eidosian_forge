import pytest
from holoviews.core.element import AttrTree
from holoviews.element.comparison import ComparisonTestCase
def test_uppercase_getitem(self):
    self.assertEqual(self.tree['A']['I'], 1)
    self.assertEqual(self.tree['B']['II'], 2)