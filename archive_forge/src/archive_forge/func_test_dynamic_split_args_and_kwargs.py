from functools import partial
import param
from holoviews import streams
from holoviews.core.operation import OperationCallable
from holoviews.core.spaces import Callable, DynamicMap, Generator
from holoviews.element import Scatter
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import contours
from ..utils import LoggingComparisonTestCase
def test_dynamic_split_args_and_kwargs(self):

    def fn(*args, **kwargs):
        return Scatter([(kwargs['x'], kwargs['y'])], label=args[0])
    xy = streams.PointerXY(x=1, y=2)
    dmap = DynamicMap(fn, kdims=['A'], streams=[xy])
    self.assertEqual(dmap['Test'], Scatter([(1, 2)], label='Test'))