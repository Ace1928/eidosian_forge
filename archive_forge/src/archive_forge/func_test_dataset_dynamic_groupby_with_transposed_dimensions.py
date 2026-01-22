import datetime
import numpy as np
import pandas as pd
from holoviews import Dataset, Dimension, HoloMap
from holoviews.core.data import concat
from holoviews.core.data.interface import DataError
from holoviews.element import Curve, Scatter
from holoviews.element.comparison import ComparisonTestCase
from holoviews.util.transform import dim
def test_dataset_dynamic_groupby_with_transposed_dimensions(self):
    dat = np.zeros((3, 5, 7))
    dataset = Dataset((range(7), range(5), range(3), dat), ['z', 'x', 'y'], 'value')
    grouped = dataset.groupby('z', kdims=['y', 'x'], dynamic=True)
    self.assertEqual(grouped[2].dimension_values(2, flat=False), dat[:, :, -1].T)