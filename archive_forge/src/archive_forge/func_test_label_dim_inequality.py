import numpy as np
import pandas as pd
from holoviews.core import Dimension, Dimensioned
from holoviews.element.comparison import ComparisonTestCase
from ..utils import LoggingComparisonTestCase
def test_label_dim_inequality(self):
    dim1 = Dimension(('test', 'label1'))
    dim2 = Dimension(('test', 'label2'))
    self.assertEqual(dim1 == dim2, False)