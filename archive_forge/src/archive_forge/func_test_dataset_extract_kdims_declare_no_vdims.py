import numpy as np
import pandas as pd
from holoviews.core.data import Dataset
from holoviews.core.data.interface import DataError
from holoviews.core.dimension import Dimension
from holoviews.core.spaces import HoloMap
from holoviews.element import Distribution, Points, Scatter
from .base import HeterogeneousColumnTests, InterfaceTests
def test_dataset_extract_kdims_declare_no_vdims(self):
    df = pd.DataFrame({'x': [1, 2, 3], 'y': [1, 2, 3], 'z': [1, 2, 3]}, columns=['x', 'y', 'z'])
    ds = Points(df, vdims=[])
    self.assertEqual(ds.kdims, [Dimension('x'), Dimension('y')])
    self.assertEqual(ds.vdims, [])