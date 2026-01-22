from holoviews import Element, HoloMap, Layout, Overlay
from holoviews.element.comparison import ComparisonTestCase
def test_layouttree_varying_label_and_values_keys(self):
    t = self.el6 + self.el7 + self.el8
    expected_keys = [('Element', 'LabelA'), ('ValA', 'LabelA'), ('ValA', 'LabelB')]
    self.assertEqual(t.keys(), expected_keys)