import numpy as np
from holoviews.core.data import Dataset
from .base import HeterogeneousColumnTests, InterfaceTests, ScalarColumnTests
def test_dataset_allow_none_values(self):
    ds = Dataset({'x': None, 'y': [0, 1]}, kdims=['x', 'y'])
    self.assertEqual(ds.dimension_values(0), np.array([None, None]))