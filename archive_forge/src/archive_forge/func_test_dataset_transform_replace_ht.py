import datetime
import numpy as np
import pandas as pd
from holoviews import Dataset, Dimension, HoloMap
from holoviews.core.data import concat
from holoviews.core.data.interface import DataError
from holoviews.element import Curve, Scatter
from holoviews.element.comparison import ComparisonTestCase
from holoviews.util.transform import dim
def test_dataset_transform_replace_ht(self):
    transformed = self.table.transform(Age=dim('Age') ** 2, Weight=dim('Weight') * 2, Height=dim('Height') / 2.0)
    expected = Dataset({'Gender': self.gender, 'Age': self.age ** 2, 'Weight': self.weight * 2, 'Height': self.height / 2.0}, kdims=self.kdims, vdims=self.vdims)
    self.assertEqual(transformed, expected)