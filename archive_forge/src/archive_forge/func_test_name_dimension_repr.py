import numpy as np
import pandas as pd
from holoviews.core import Dimension, Dimensioned
from holoviews.element.comparison import ComparisonTestCase
from ..utils import LoggingComparisonTestCase
def test_name_dimension_repr(self):
    dim = Dimension('test')
    self.assertEqual(repr(dim), "Dimension('test')")