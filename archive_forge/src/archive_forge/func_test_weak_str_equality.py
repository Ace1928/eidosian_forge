import numpy as np
import pandas as pd
from holoviews.core import Dimension, Dimensioned
from holoviews.element.comparison import ComparisonTestCase
from ..utils import LoggingComparisonTestCase
def test_weak_str_equality(self):
    dim1 = Dimension('test', cyclic=True, unit='m', type=float)
    dim2 = Dimension('test', cyclic=False, unit='km', type=int)
    self.assertEqual(dim1 == str(dim2), True)