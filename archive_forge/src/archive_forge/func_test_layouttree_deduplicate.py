from holoviews import Element, HoloMap, Layout, Overlay
from holoviews.element.comparison import ComparisonTestCase
def test_layouttree_deduplicate(self):
    for i in range(2, 10):
        l = Layout([Element([], label='0') for _ in range(i)])
        self.assertEqual(len(l), i)