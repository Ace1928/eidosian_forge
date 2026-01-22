from holoviews import Element, HoloMap, Layout, Overlay
from holoviews.element.comparison import ComparisonTestCase
def test_layouttree_quadruple_2(self):
    t = self.el6 + self.el6 + self.el6 + self.el6
    self.assertEqual(t.keys(), [('Element', 'LabelA', 'I'), ('Element', 'LabelA', 'II'), ('Element', 'LabelA', 'III'), ('Element', 'LabelA', 'IV')])