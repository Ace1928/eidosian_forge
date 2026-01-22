from holoviews import Bounds, Box, Contours, Ellipse, Path
from holoviews.element.comparison import ComparisonTestCase
def test_contours_unequal(self):
    try:
        self.assertEqual(self.contours1, self.contours2)
    except AssertionError as e:
        if not str(e).startswith('Contours not almost equal to 6 decimals'):
            raise self.failureException('Contours mismatch error not raised.')