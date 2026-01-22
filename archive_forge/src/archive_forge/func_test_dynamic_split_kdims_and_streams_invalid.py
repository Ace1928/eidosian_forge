from functools import partial
import param
from holoviews import streams
from holoviews.core.operation import OperationCallable
from holoviews.core.spaces import Callable, DynamicMap, Generator
from holoviews.element import Scatter
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import contours
from ..utils import LoggingComparisonTestCase
def test_dynamic_split_kdims_and_streams_invalid(self):

    def fn(x=1, y=2, B='default'):
        return Scatter([(x, y)], label=B)
    xy = streams.PointerXY(x=1, y=2)
    regexp = "Callback 'fn' signature over (.+?) does not accommodate required kdims"
    with self.assertRaisesRegex(KeyError, regexp):
        DynamicMap(fn, kdims=['A'], streams=[xy])