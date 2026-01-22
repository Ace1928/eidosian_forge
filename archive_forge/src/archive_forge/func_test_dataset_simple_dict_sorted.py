from unittest import SkipTest
import numpy as np
from holoviews.core.data import Dataset
from .base import HomogeneousColumnTests, InterfaceTests
def test_dataset_simple_dict_sorted(self):
    dataset = Dataset({2: 2, 1: 1, 3: 3}, kdims=['x'], vdims=['y'])
    self.assertEqual(dataset, Dataset([(i, i) for i in range(1, 4)], kdims=['x'], vdims=['y']))