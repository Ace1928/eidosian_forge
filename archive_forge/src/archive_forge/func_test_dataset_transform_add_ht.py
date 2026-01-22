import datetime
import numpy as np
import pandas as pd
from holoviews import Dataset, Dimension, HoloMap
from holoviews.core.data import concat
from holoviews.core.data.interface import DataError
from holoviews.element import Curve, Scatter
from holoviews.element.comparison import ComparisonTestCase
from holoviews.util.transform import dim
def test_dataset_transform_add_ht(self):
    transformed = self.table.transform(combined=dim('Age') * dim('Weight'))
    expected = Dataset({'Gender': self.gender, 'Age': self.age, 'Weight': self.weight, 'Height': self.height, 'combined': self.age * self.weight}, kdims=self.kdims, vdims=self.vdims + ['combined'])
    self.assertEqual(transformed, expected)