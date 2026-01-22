from holoviews import Element, HoloMap, Layout, Overlay
from holoviews.element.comparison import ComparisonTestCase
def test_overlay_varying_label_keys(self):
    t = self.el1 * self.el6
    self.assertEqual(t.keys(), [('Element', 'I'), ('Element', 'LabelA')])