from holoviews import Curve, DynamicMap, HoloMap
from holoviews.core.traversal import unique_dimkeys
from holoviews.element.comparison import ComparisonTestCase
def test_unique_keys_no_overlap_dynamicmap_initialized(self):
    dmap1 = DynamicMap(lambda A: Curve(range(10)), kdims=['A'])
    dmap2 = DynamicMap(lambda B: Curve(range(10)), kdims=['B'])
    dmap1[0]
    dmap2[1]
    dims, keys = unique_dimkeys(dmap1 + dmap2)
    self.assertEqual(dims, dmap1.kdims + dmap2.kdims)
    self.assertEqual(keys, [(0, 1)])