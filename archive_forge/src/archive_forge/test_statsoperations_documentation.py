from unittest import SkipTest
import numpy as np
from holoviews import Area, Bivariate, Contours, Distribution, Image, Polygons
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation.stats import bivariate_kde, univariate_kde

    Tests for the various timeseries operations including rolling,
    resample and rolling_outliers_std.
    