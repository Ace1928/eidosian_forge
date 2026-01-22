import numpy as np
from holoviews.core.data import Dataset
from .base import HeterogeneousColumnTests, InterfaceTests, ScalarColumnTests
def test_dataset_ignore_non_dimensions(self):
    ds = Dataset({'x': [0, 1], 'y': [1, 2], 'ignore_scalar': 1, 'ignore_array': np.array([2, 3]), 'ignore_None': None}, kdims=['x', 'y'])
    ds2 = Dataset({'x': [0, 1], 'y': [1, 2]}, kdims=['x', 'y'])
    self.assertEqual(ds, ds2)