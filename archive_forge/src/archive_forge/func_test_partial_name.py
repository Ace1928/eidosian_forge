from functools import partial
import param
from holoviews import streams
from holoviews.core.operation import OperationCallable
from holoviews.core.spaces import Callable, DynamicMap, Generator
from holoviews.element import Scatter
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import contours
from ..utils import LoggingComparisonTestCase
def test_partial_name(self):
    cb = Callable(partial(lambda x, y: x, y=4))
    self.assertEqual(cb.name.startswith('functools.partial('), True)