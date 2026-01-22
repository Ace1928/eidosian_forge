from holoviews import Element, HoloMap, Layout, Overlay
from holoviews.element.comparison import ComparisonTestCase
def test_layouttree_varying_label_keys2(self):
    t = self.el7 + self.el8
    self.assertEqual(t.keys(), [('ValA', 'LabelA'), ('ValA', 'LabelB')])