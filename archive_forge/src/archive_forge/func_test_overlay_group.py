from holoviews import Element, HoloMap, Layout, Overlay
from holoviews.element.comparison import ComparisonTestCase
def test_overlay_group(self):
    t1 = self.el1 * self.el2
    t2 = Overlay(list(t1.relabel(group='NewValue', depth=1)))
    self.assertEqual(t2.keys(), [('NewValue', 'I'), ('NewValue', 'II')])