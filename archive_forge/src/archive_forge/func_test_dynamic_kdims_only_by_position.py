from functools import partial
import param
from holoviews import streams
from holoviews.core.operation import OperationCallable
from holoviews.core.spaces import Callable, DynamicMap, Generator
from holoviews.element import Scatter
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import contours
from ..utils import LoggingComparisonTestCase
def test_dynamic_kdims_only_by_position(self):

    def fn(A, B):
        return Scatter([(B, 2)], label=A)
    dmap = DynamicMap(fn, kdims=['A-dim', 'B-dim'])
    self.assertEqual(dmap['Test', 1], Scatter([(1, 2)], label='Test'))