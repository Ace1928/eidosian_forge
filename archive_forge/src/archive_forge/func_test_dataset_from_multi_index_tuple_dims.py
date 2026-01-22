import unittest
from unittest import SkipTest
import numpy as np
import pandas as pd
from packaging.version import Version
from holoviews.core.data import Dataset
from holoviews.core.util import pandas_version
from holoviews.util.transform import dim
from .test_pandasinterface import BasePandasInterfaceTests
def test_dataset_from_multi_index_tuple_dims(self):
    raise SkipTest('Temporarily skipped')
    df = pd.DataFrame({'x': np.arange(10), 'y': np.arange(10), 'z': np.random.rand(10)})
    ddf = dd.from_pandas(df, 1)
    ds = Dataset(ddf.groupby(['x', 'y']).mean(), [('x', 'X'), ('y', 'Y')])
    self.assertEqual(ds, Dataset(df, [('x', 'X'), ('y', 'Y')]))