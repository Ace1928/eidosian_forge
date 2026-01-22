import pytest
from holoviews.core.element import AttrTree
from holoviews.element.comparison import ComparisonTestCase
def test_number_getitem_key_error(self):
    with self.assertRaises(KeyError):
        self.tree['2']