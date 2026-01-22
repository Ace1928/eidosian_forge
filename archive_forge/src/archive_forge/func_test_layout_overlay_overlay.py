from holoviews import Element, HoloMap, Layout, Overlay
from holoviews.element.comparison import ComparisonTestCase
def test_layout_overlay_overlay(self):
    t = (self.el1 + self.el2) * (self.el3 * self.el4)
    self.assertEqual(t, Layout([self.el1 * self.el3 * self.el4, self.el2 * self.el3 * self.el4]))