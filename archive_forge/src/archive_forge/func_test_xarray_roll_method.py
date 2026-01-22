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
def test_xarray_roll_method(self):
    expr = dim('z').xr.roll({'x': 1}, roll_coords=False)
    self.assert_apply_xarray(expr, self.dataset_xarray.data.z.roll({'x': 1}, roll_coords=False))