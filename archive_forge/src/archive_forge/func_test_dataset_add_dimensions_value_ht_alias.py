import datetime
import numpy as np
import pandas as pd
from holoviews import Dataset, Dimension, HoloMap
from holoviews.core.data import concat
from holoviews.core.data.interface import DataError
from holoviews.element import Curve, Scatter
from holoviews.element.comparison import ComparisonTestCase
from holoviews.util.transform import dim
def test_dataset_add_dimensions_value_ht_alias(self):
    table = self.dataset_ht.add_dimension(('z', 'Z'), 1, 0)
    self.assertEqual(table.kdims[1], 'z')
    self.compare_arrays(table.dimension_values('z'), np.zeros(table.shape[0]))