import pickle
import warnings
from unittest import skipIf
import numpy as np
import pandas as pd
import param
import holoviews as hv
from holoviews.core.data import Dataset
from holoviews.element.comparison import ComparisonTestCase
from holoviews.util.transform import dim
def test_pandas_namespace_accessor_repr(self):
    self.assertEqual(repr(dim('date').df.dt.year), "dim('date').pd.dt.year")