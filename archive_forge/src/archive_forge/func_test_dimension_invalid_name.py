import numpy as np
import pandas as pd
from holoviews.core import Dimension, Dimensioned
from holoviews.element.comparison import ComparisonTestCase
from ..utils import LoggingComparisonTestCase
def test_dimension_invalid_name(self):
    regexp = 'Dimension name must only be passed as the positional argument'
    with self.assertRaisesRegex(KeyError, regexp):
        Dimension('test', name='something else')