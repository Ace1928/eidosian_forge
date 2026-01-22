from holoviews import Element, HoloMap, Layout, Overlay
from holoviews.element.comparison import ComparisonTestCase
def test_layout_overlay_element_reverse(self):
    t = self.el3 * (self.el1 + self.el2)
    self.assertEqual(t, Layout([self.el3 * self.el1, self.el3 * self.el2]))