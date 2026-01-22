import param
from holoviews.core.operation import Operation
from holoviews.element import Curve
from holoviews.element.comparison import ComparisonTestCase
from holoviews.streams import Params, Stream
def test_element_dynamic_with_streams(self):
    curve = Curve([1, 2, 3])
    applied = Operation(curve, dynamic=True, streams=[Stream])
    self.assertEqual(len(applied.streams), 1)
    self.assertIsInstance(applied.streams[0], Stream)
    self.assertEqual(applied[()], curve)