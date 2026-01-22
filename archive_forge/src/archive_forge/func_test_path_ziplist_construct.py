import numpy as np
import pandas as pd
import param
from holoviews import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element.path import BaseShape
def test_path_ziplist_construct(self):
    self.assertEqual(Path([list(zip(self.xs, self.sin)), list(zip(self.xs, self.cos))]), self.path)