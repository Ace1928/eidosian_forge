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
@xr_skip
def test_xarray_namespace_method_repr(self):
    self.assertEqual(repr(dim('date').xr.quantile(0.95)), "dim('date').xr.quantile(0.95)")