from unittest import SkipTest
import numpy as np
import pandas as pd
from holoviews import Curve, Dataset, Dimension, Distribution, Scatter
from holoviews.core import Apply, Redim
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import function, histogram
def test_distribution_dataset(self):
    self.assertEqual(self.distribution.dataset, self.ds)
    self.assertEqual(self.distribution.pipeline(self.distribution.dataset), self.distribution)