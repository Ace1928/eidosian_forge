import numpy as np
import pandas as pd
from holoviews.core import Dimension, Dimensioned
from holoviews.element.comparison import ComparisonTestCase
from ..utils import LoggingComparisonTestCase
def test_simple_str_inequality(self):
    dim1 = Dimension('test1')
    dim2 = Dimension('test2')
    self.assertEqual(dim1 == str(dim2), False)