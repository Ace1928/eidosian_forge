from unittest import skipIf
import param
from holoviews.core import DynamicMap, GridSpace, HoloMap, NdOverlay, Overlay
from holoviews.element import Points
from holoviews.element.comparison import ComparisonTestCase
from holoviews.streams import PlotSize, RangeXY, Stream
def test_decollate_layout_streams(self):
    layout = self.dmap_xy + self.dmap_z
    decollated = layout.decollate()
    self.assertIsInstance(decollated, DynamicMap)
    self.assertEqual(decollated.kdims, [])
    decollated.streams[0].event(x=1.0, y=2.0)
    decollated.streams[1].event(z=3.0)
    self.assertEqual(decollated[()], Points([1.0, 2.0]) + Points([3.0, 3.0]))
    self.assertEqual(decollated.callback.callable(dict(x=1.0, y=2.0), dict(z=3.0)), Points([1.0, 2.0]) + Points([3.0, 3.0]))