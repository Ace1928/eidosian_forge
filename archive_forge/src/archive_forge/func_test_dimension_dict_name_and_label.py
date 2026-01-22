import numpy as np
import pandas as pd
from holoviews.core import Dimension, Dimensioned
from holoviews.element.comparison import ComparisonTestCase
from ..utils import LoggingComparisonTestCase
def test_dimension_dict_name_and_label(self):
    dim = Dimension(dict(name='test', label='A test'))
    self.assertEqual(dim.name, 'test')
    self.assertEqual(dim.label, 'A test')