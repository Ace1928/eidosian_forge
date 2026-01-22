import numpy as np
import pandas as pd
from holoviews.core import Dimension, Dimensioned
from holoviews.element.comparison import ComparisonTestCase
from ..utils import LoggingComparisonTestCase
def test_dimension_values_list2(self):
    dim = Dimension('test', values=self.values2)
    self.assertEqual(dim.values, self.values2)