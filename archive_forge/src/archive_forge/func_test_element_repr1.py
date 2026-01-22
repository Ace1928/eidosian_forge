from holoviews import Curve, Element, Layout, Overlay, Store
from holoviews.core.pprint import PrettyPrinter
from holoviews.element.comparison import ComparisonTestCase
from .test_dimensioned import CustomBackendTestCase, ExampleElement
def test_element_repr1(self):
    r = PrettyPrinter.pprint(self.element1)
    self.assertEqual(r, ':Element')