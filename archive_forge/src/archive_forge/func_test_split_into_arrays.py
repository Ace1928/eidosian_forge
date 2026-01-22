import logging
import numpy as np
import pandas as pd
from param import get_logger
from holoviews.core.data import Dataset, MultiInterface
from holoviews.element import Path, Points, Polygons
from holoviews.element.comparison import ComparisonTestCase
def test_split_into_arrays(self):
    arrays = [np.column_stack([np.arange(i, i + 2), np.arange(i, i + 2)]) for i in range(2)]
    mds = Path(arrays, kdims=['x', 'y'], datatype=[self.datatype])
    self.assertIs(mds.interface, self.interface)
    for arr1, arr2 in zip(mds.split(datatype='array'), arrays):
        self.assertEqual(arr1, arr2)