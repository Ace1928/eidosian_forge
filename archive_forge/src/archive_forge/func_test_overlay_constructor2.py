from holoviews import Element, HoloMap, Layout, Overlay
from holoviews.element.comparison import ComparisonTestCase
def test_overlay_constructor2(self):
    t = Overlay([self.el8])
    self.assertEqual(t.keys(), [('ValA', 'LabelB')])