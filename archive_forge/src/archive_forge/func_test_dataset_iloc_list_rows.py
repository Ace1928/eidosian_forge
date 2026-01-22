import datetime
import numpy as np
import pandas as pd
from holoviews import Dataset, Dimension, HoloMap
from holoviews.core.data import concat
from holoviews.core.data.interface import DataError
from holoviews.element import Curve, Scatter
from holoviews.element.comparison import ComparisonTestCase
from holoviews.util.transform import dim
def test_dataset_iloc_list_rows(self):
    sliced = self.table.iloc[[0, 2]]
    table = Dataset({'Gender': self.gender[[0, 2]], 'Age': self.age[[0, 2]], 'Weight': self.weight[[0, 2]], 'Height': self.height[[0, 2]]}, kdims=self.kdims, vdims=self.vdims)
    self.assertEqual(sliced, table)