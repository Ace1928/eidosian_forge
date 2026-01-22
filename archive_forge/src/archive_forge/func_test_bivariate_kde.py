from unittest import SkipTest
import numpy as np
from holoviews import Area, Bivariate, Contours, Distribution, Image, Polygons
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation.stats import bivariate_kde, univariate_kde
def test_bivariate_kde(self):
    kde = bivariate_kde(self.bivariate, n_samples=2, x_range=(0, 4), y_range=(0, 4), contours=False)
    img = Image(np.array([[0.021315, 0.021315], [0.021315, 0.021315]]), bounds=(-2, -2, 6, 6), vdims=['Density'])
    self.assertEqual(kde, img)