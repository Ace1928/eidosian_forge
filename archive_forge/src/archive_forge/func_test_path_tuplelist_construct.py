import numpy as np
import pandas as pd
import param
from holoviews import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element.path import BaseShape
def test_path_tuplelist_construct(self):
    self.assertEqual(Path([(self.xs, self.sin), (self.xs, self.cos)]), self.path)