import datetime
import numpy as np
import pandas as pd
from holoviews import Dataset, Dimension, HoloMap
from holoviews.core.data import concat
from holoviews.core.data.interface import DataError
from holoviews.element import Curve, Scatter
from holoviews.element.comparison import ComparisonTestCase
from holoviews.util.transform import dim
def test_dataset_reduce_ht(self):
    reduced = Dataset({'Age': self.age, 'Weight': self.weight, 'Height': self.height}, kdims=self.kdims[1:], vdims=self.vdims)
    self.assertEqual(self.table.reduce(['Gender'], np.mean), reduced)