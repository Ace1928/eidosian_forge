import logging
import numpy as np
import pandas as pd
from param import get_logger
from holoviews.core.data import Dataset, MultiInterface
from holoviews.element import Path, Points, Polygons
from holoviews.element.comparison import ComparisonTestCase
def test_dict_dataset_add_dimension_scalar(self):
    arrays = [{'x': np.arange(i, i + 2), 'y': np.arange(i, i + 2)} for i in range(2)]
    mds = Path(arrays, kdims=['x', 'y'], datatype=[self.datatype]).add_dimension('A', 0, 'Scalar', True)
    self.assertIs(mds.interface, self.interface)
    self.assertEqual(mds, Path([dict(arrays[i], A='Scalar') for i in range(2)], ['x', 'y'], 'A', datatype=['multitabular']))