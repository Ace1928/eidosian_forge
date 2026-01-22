from unittest import skipIf
import param
from holoviews.core import DynamicMap, GridSpace, HoloMap, NdOverlay, Overlay
from holoviews.element import Points
from holoviews.element.comparison import ComparisonTestCase
from holoviews.streams import PlotSize, RangeXY, Stream
def test_decollate_layout_kdims(self):
    layout = self.dmap_ab + self.dmap_b
    decollated = layout.decollate()
    self.assertIsInstance(decollated, DynamicMap)
    self.assertEqual(decollated.kdims, self.dmap_ab.kdims)
    self.assertEqual(decollated[2, 3], Points([2, 3]) + Points([3, 3]))
    self.assertEqual(decollated.callback.callable(2, 3), Points([2, 3]) + Points([3, 3]))