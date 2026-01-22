import numpy as np
import pandas as pd
from holoviews.core import Dimension, Dimensioned
from holoviews.element.comparison import ComparisonTestCase
from ..utils import LoggingComparisonTestCase
def test_simple_values_clone(self):
    dim = Dimension('test', values=[1, 2, 3])
    self.assertEqual(dim.values, [1, 2, 3])
    clone = dim.clone(values=[4, 5, 6])
    self.assertEqual(clone.name, 'test')
    self.assertEqual(clone.values, [4, 5, 6])