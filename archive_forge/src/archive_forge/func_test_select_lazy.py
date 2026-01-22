import datetime as dt
from itertools import product
from unittest import SkipTest
import numpy as np
import pandas as pd
from holoviews.core.data import Dataset
from holoviews.core.data.interface import DataError
from holoviews.core.util import date_range
from holoviews.element import HSV, RGB, Curve, Image
from holoviews.util.transform import dim
from .base import (
from .test_imageinterface import (
def test_select_lazy(self):
    import dask.array as da
    arr = da.from_array(np.arange(1, 12), 3)
    ds = Dataset({'x': range(11), 'y': arr}, 'x', 'y')
    self.assertIsInstance(ds.select(x=(0, 5)).data['y'], da.Array)