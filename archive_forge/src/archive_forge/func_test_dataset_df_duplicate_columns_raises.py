import numpy as np
import pandas as pd
from holoviews.core.data import Dataset
from holoviews.core.data.interface import DataError
from holoviews.core.dimension import Dimension
from holoviews.core.spaces import HoloMap
from holoviews.element import Distribution, Points, Scatter
from .base import HeterogeneousColumnTests, InterfaceTests
def test_dataset_df_duplicate_columns_raises(self):
    df = pd.DataFrame(np.random.randint(-100, 100, size=(100, 2)), columns=list('AB'))
    with self.assertRaises(DataError):
        Dataset(df[['A', 'A']])