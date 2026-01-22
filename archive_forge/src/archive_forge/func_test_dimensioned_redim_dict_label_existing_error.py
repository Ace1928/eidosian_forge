import numpy as np
import pandas as pd
from holoviews.core import Dimension, Dimensioned
from holoviews.element.comparison import ComparisonTestCase
from ..utils import LoggingComparisonTestCase
def test_dimensioned_redim_dict_label_existing_error(self):
    dimensioned = Dimensioned('Arbitrary Data', kdims=[('x', 'Test1')])
    with self.assertRaisesRegex(ValueError, 'Cannot override an existing Dimension label'):
        dimensioned.redim.label(x='Test2')