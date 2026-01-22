from holoviews import Element, HoloMap, Layout, Overlay
from holoviews.element.comparison import ComparisonTestCase
def test_layout_integer_index(self):
    t = self.el1 + self.el2
    self.assertEqual(t[0], self.el1)
    self.assertEqual(t[1], self.el2)