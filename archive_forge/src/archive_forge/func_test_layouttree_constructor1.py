from holoviews import Element, HoloMap, Layout, Overlay
from holoviews.element.comparison import ComparisonTestCase
def test_layouttree_constructor1(self):
    t = Layout([self.el1])
    self.assertEqual(t.keys(), [('Element', 'I')])