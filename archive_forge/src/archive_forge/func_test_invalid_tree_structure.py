from holoviews import Element, HoloMap, Layout, Overlay
from holoviews.element.comparison import ComparisonTestCase
def test_invalid_tree_structure(self):
    try:
        (self.el1 + self.el2) * (self.el1 + self.el2)
    except TypeError as e:
        self.assertEqual(str(e), "unsupported operand type(s) for *: 'Layout' and 'Layout'")