import numpy as np
from holoviews import Box, Dataset, Ellipse, Path, Polygons
from holoviews.core.data.interface import DataError
from holoviews.element.comparison import ComparisonTestCase
def test_ellipse_simple_constructor_squashed_aspect(self):
    ellipse = Ellipse(0, 0, 2, aspect=0.5, samples=6)
    self.assertEqual(np.allclose(ellipse.data[0], self.squashed), True)