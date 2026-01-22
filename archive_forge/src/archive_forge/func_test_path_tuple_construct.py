import numpy as np
import pandas as pd
import param
from holoviews import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element.path import BaseShape
def test_path_tuple_construct(self):
    self.assertEqual(Path((self.xs, np.column_stack((self.sin, self.cos)))), self.path)