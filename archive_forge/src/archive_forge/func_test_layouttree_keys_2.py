from holoviews import Element, HoloMap, Layout, Overlay
from holoviews.element.comparison import ComparisonTestCase
def test_layouttree_keys_2(self):
    t = Layout([self.el1, self.el2])
    self.assertEqual(t.keys(), [('Element', 'I'), ('Element', 'II')])