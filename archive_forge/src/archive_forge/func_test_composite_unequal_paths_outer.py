from holoviews import Element
from holoviews.element.comparison import ComparisonTestCase
def test_composite_unequal_paths_outer(self):
    t1 = self.el1 * self.el2 + (self.el1 * self.el2).relabel(group='ValA')
    t2 = self.el1 * self.el2 + self.el1 * self.el3
    try:
        self.assertEqual(t1, t2)
    except AssertionError as e:
        self.assertEqual(str(e), 'Layouts have mismatched paths.')