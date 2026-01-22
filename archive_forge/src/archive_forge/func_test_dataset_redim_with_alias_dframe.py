import datetime
import numpy as np
import pandas as pd
from holoviews import Dataset, Dimension, HoloMap
from holoviews.core.data import concat
from holoviews.core.data.interface import DataError
from holoviews.element import Curve, Scatter
from holoviews.element.comparison import ComparisonTestCase
from holoviews.util.transform import dim
def test_dataset_redim_with_alias_dframe(self):
    test_df = pd.DataFrame({'x': range(10), 'y': range(0, 20, 2)})
    dataset = Dataset(test_df, kdims=[('x', 'X-label')], vdims=['y'])
    redim_df = pd.DataFrame({'X': range(10), 'y': range(0, 20, 2)})
    dataset_redim = Dataset(redim_df, kdims=['X'], vdims=['y'])
    self.assertEqual(dataset.redim(**{'X-label': 'X'}), dataset_redim)
    self.assertEqual(dataset.redim(x='X'), dataset_redim)