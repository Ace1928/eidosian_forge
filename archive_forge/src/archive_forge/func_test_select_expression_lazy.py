import unittest
from unittest import SkipTest
import numpy as np
import pandas as pd
from packaging.version import Version
from holoviews.core.data import Dataset
from holoviews.core.util import pandas_version
from holoviews.util.transform import dim
from .test_pandasinterface import BasePandasInterfaceTests
def test_select_expression_lazy(self):
    df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [10, 10, 11, 11, 10]})
    ddf = dd.from_pandas(df, npartitions=2)
    ds = Dataset(ddf)
    new_ds = ds.select(selection_expr=dim('b') == 10)
    self.assertIsInstance(new_ds.data, dd.DataFrame)
    self.assertEqual(new_ds.data.compute(), df[df.b == 10])