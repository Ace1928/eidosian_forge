import datetime
import numpy as np
import pandas as pd
from holoviews import Dataset, Dimension, HoloMap
from holoviews.core.data import concat
from holoviews.core.data.interface import DataError
from holoviews.element import Curve, Scatter
from holoviews.element.comparison import ComparisonTestCase
from holoviews.util.transform import dim
def test_dataset_select_rows_gender_male_alias(self):
    row = self.alias_table.select(Gender='M')
    alias_row = self.alias_table.select(gender='M')
    indexed = Dataset({'gender': ['M', 'M'], 'age': [10, 16], 'weight': [15, 18], 'height': [0.8, 0.6]}, kdims=self.alias_kdims, vdims=self.alias_vdims)
    self.assertEqual(row, indexed)
    self.assertEqual(alias_row, indexed)