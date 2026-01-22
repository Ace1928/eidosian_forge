from holoviews import Element, HoloMap, Layout, Overlay
from holoviews.element.comparison import ComparisonTestCase
def test_overlay_constructor_retains_custom_path(self):
    overlay = Overlay([('Custom', self.el1)])
    paths = Overlay([overlay, self.el2]).keys()
    self.assertEqual(paths, [('Custom', 'I'), ('Element', 'I')])