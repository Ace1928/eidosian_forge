from functools import partial
import param
from holoviews import streams
from holoviews.core.operation import OperationCallable
from holoviews.core.spaces import Callable, DynamicMap, Generator
from holoviews.element import Scatter
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import contours
from ..utils import LoggingComparisonTestCase
def test_dynamic_split_mismatched_kdims_invalid(self):

    def fn(x, y, B):
        return Scatter([(x, y)], label=B)
    xy = streams.PointerXY(x=1, y=2)
    regexp = 'Unmatched positional kdim arguments only allowed at the start of the signature'
    with self.assertRaisesRegex(KeyError, regexp):
        DynamicMap(fn, kdims=['A'], streams=[xy])