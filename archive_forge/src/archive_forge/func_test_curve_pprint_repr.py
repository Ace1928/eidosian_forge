from holoviews import Curve, Element, Layout, Overlay, Store
from holoviews.core.pprint import PrettyPrinter
from holoviews.element.comparison import ComparisonTestCase
from .test_dimensioned import CustomBackendTestCase, ExampleElement
def test_curve_pprint_repr(self):
    expected = "':Curve   [x]   (y)'"
    r = PrettyPrinter.pprint(Curve([1, 2, 3]))
    self.assertEqual(repr(r), expected)