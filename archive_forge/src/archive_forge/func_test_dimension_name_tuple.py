import numpy as np
import pandas as pd
from holoviews.core import Dimension, Dimensioned
from holoviews.element.comparison import ComparisonTestCase
from ..utils import LoggingComparisonTestCase
def test_dimension_name_tuple(self):
    dim = Dimension(('test', 'A test'))
    self.assertEqual(dim.name, 'test')