from holoviews import Element, HoloMap, Layout, Overlay
from holoviews.element.comparison import ComparisonTestCase
def test_layout_constructor_retains_custom_path(self):
    layout = Layout([('Custom', self.el1)])
    paths = Layout([layout, self.el2]).keys()
    self.assertEqual(paths, [('Custom', 'I'), ('Element', 'I')])