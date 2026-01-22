from holoviews import Element, HoloMap, Layout, Overlay
from holoviews.element.comparison import ComparisonTestCase
def test_composite1(self):
    t = self.el1 * self.el2 + self.el1 * self.el2
    self.assertEqual(t.keys(), [('Overlay', 'I'), ('Overlay', 'II')])