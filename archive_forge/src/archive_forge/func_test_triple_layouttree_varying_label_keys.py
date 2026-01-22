from holoviews import Element, HoloMap, Layout, Overlay
from holoviews.element.comparison import ComparisonTestCase
def test_triple_layouttree_varying_label_keys(self):
    t = self.el1 + self.el6 + self.el2
    expected_keys = [('Element', 'I'), ('Element', 'LabelA'), ('Element', 'II')]
    self.assertEqual(t.keys(), expected_keys)