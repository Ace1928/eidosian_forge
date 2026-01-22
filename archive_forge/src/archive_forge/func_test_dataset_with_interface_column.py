import numpy as np
import pandas as pd
from holoviews.core.data import Dataset
from holoviews.core.data.interface import DataError
from holoviews.core.dimension import Dimension
from holoviews.core.spaces import HoloMap
from holoviews.element import Distribution, Points, Scatter
from .base import HeterogeneousColumnTests, InterfaceTests
def test_dataset_with_interface_column(self):
    df = pd.DataFrame([1], columns=['interface'])
    ds = Dataset(df)
    self.assertEqual(list(ds.data.columns), ['interface'])