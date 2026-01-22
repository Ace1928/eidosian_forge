import numpy as np
import pandas as pd
import param
from holoviews import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element.path import BaseShape
def test_path_string_signature(self):
    path = Path([], ['a', 'b'])
    self.assertEqual(path.kdims, [Dimension('a'), Dimension('b')])