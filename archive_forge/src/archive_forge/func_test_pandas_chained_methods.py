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
def test_pandas_chained_methods(self):
    expr = dim('int').df.rolling(1).mean()
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'Passing additional kwargs to Rolling.mean')
        self.assert_apply(expr, self.linear_ints.rolling(1).mean())