from functools import partial
import param
from holoviews import streams
from holoviews.core.operation import OperationCallable
from holoviews.core.spaces import Callable, DynamicMap, Generator
from holoviews.element import Scatter
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import contours
from ..utils import LoggingComparisonTestCase
def test_callable_paramfunc_instance_argspec(self):
    self.assertEqual(Callable(ParamFunc.instance()).argspec.args, ['a'])
    self.assertEqual(Callable(ParamFunc.instance()).argspec.keywords, 'params')
    self.assertEqual(Callable(ParamFunc.instance()).argspec.varargs, None)