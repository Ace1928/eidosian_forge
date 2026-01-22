from unittest import skipIf
import param
from holoviews.core import DynamicMap, GridSpace, HoloMap, NdOverlay, Overlay
from holoviews.element import Points
from holoviews.element.comparison import ComparisonTestCase
from holoviews.streams import PlotSize, RangeXY, Stream
def test_decollate_overlay_of_dmaps(self):
    overlay = Overlay([DynamicMap(lambda z: Points([z, z]), streams=[Z()]), DynamicMap(lambda z: Points([z, z]), streams=[Z()]), DynamicMap(lambda z: Points([z, z]), streams=[Z()])])
    decollated = overlay.decollate()
    self.assertIsInstance(decollated, DynamicMap)
    self.assertEqual(len(decollated.streams), 3)
    expected = Overlay([Points([1.0, 1.0]), Points([2.0, 2.0]), Points([3.0, 3.0])])
    decollated.streams[0].event(z=1.0)
    decollated.streams[1].event(z=2.0)
    decollated.streams[2].event(z=3.0)
    result = decollated[()]
    self.assertEqual(expected, result)
    result = decollated.callback.callable(dict(z=1.0), dict(z=2.0), dict(z=3.0))
    self.assertEqual(expected, result)