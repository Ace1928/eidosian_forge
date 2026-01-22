import unittest
from unittest import SkipTest
import numpy as np
import pandas as pd
from packaging.version import Version
from holoviews.core.data import Dataset
from holoviews.core.util import pandas_version
from holoviews.util.transform import dim
from .test_pandasinterface import BasePandasInterfaceTests
@unittest.skipIf(pandas_version >= Version('2.0'), reason='Not supported yet, https://github.com/dask/dask/issues/9913')
def test_dataset_aggregate_ht_alias(self):
    super().test_dataset_aggregate_ht_alias()