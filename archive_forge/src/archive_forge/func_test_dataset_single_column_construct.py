import numpy as np
import pandas as pd
from holoviews.core.data import Dataset
from holoviews.core.data.interface import DataError
from holoviews.core.dimension import Dimension
from holoviews.core.spaces import HoloMap
from holoviews.element import Distribution, Points, Scatter
from .base import HeterogeneousColumnTests, InterfaceTests
def test_dataset_single_column_construct(self):
    ds = Scatter(pd.DataFrame([1, 2, 3], columns=['A']))
    self.assertEqual(ds, Scatter(([0, 1, 2], [1, 2, 3]), 'index', 'A'))