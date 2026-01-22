from unittest import SkipTest
import numpy as np
from holoviews import Area, Bivariate, Contours, Distribution, Image, Polygons
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation.stats import bivariate_kde, univariate_kde
def test_univariate_kde_flat_distribution(self):
    dist = Distribution([1, 1, 1])
    kde = univariate_kde(dist, n_samples=5, bin_range=(0, 4))
    area = Area([], 'Value', ('Value_density', 'Density'))
    self.assertEqual(kde, area)