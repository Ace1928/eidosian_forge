import numpy as np
import pandas as pd
from holoviews.core import Dimension, Dimensioned
from holoviews.element.comparison import ComparisonTestCase
from ..utils import LoggingComparisonTestCase
def test_name_dimension_repr_eval_equality(self):
    dim = Dimension('test')
    self.assertEqual(eval(repr(dim)) == dim, True)