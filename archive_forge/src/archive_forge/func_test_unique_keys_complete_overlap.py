from holoviews import Curve, DynamicMap, HoloMap
from holoviews.core.traversal import unique_dimkeys
from holoviews.element.comparison import ComparisonTestCase
def test_unique_keys_complete_overlap(self):
    hmap1 = HoloMap({i: Curve(range(10)) for i in range(5)})
    hmap2 = HoloMap({i: Curve(range(10)) for i in range(3, 10)})
    dims, keys = unique_dimkeys(hmap1 + hmap2)
    self.assertEqual(hmap1.kdims, dims)
    self.assertEqual(keys, [(i,) for i in range(10)])