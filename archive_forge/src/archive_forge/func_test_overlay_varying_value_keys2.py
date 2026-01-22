from holoviews import Element, HoloMap, Layout, Overlay
from holoviews.element.comparison import ComparisonTestCase
def test_overlay_varying_value_keys2(self):
    t = self.el4 * self.el5
    self.assertEqual(t.keys(), [('ValA', 'I'), ('ValB', 'I')])