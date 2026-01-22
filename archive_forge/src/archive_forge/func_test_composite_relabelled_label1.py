from holoviews import Element, HoloMap, Layout, Overlay
from holoviews.element.comparison import ComparisonTestCase
def test_composite_relabelled_label1(self):
    t = self.el1 * self.el2 + (self.el1 * self.el2).relabel(group='Val1', label='Label2')
    self.assertEqual(t.keys(), [('Overlay', 'I'), ('Val1', 'Label2')])