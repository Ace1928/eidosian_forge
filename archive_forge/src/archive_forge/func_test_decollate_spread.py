from unittest import skipIf
import param
from holoviews.core import DynamicMap, GridSpace, HoloMap, NdOverlay, Overlay
from holoviews.element import Points
from holoviews.element.comparison import ComparisonTestCase
from holoviews.streams import PlotSize, RangeXY, Stream
@datashade_skip
def test_decollate_spread(self):
    decollated = self.dmap_spread_points.decollate()
    self.assertIsInstance(decollated, DynamicMap)
    self.assertEqual([PlotSize, RangeXY, PX], [type(s) for s in decollated.streams])
    self.px_stream.event(px=3)
    plot_size, range_xy = self.dmap_spread_points.callback.inputs[0].streams
    plot_size.event(width=250, height=300)
    range_xy.event(x_range=(0, 10), y_range=(0, 15))
    expected = self.dmap_spread_points[()]
    result = decollated.callback.callable({'width': 250, 'height': 300}, {'x_range': (0, 10), 'y_range': (0, 15)}, {'px': 3})
    self.assertEqual(expected, result)