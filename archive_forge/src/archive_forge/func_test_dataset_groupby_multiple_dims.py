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
def test_dataset_groupby_multiple_dims(self):
    dataset = Dataset((range(8), range(8), range(8), range(8), da.from_array(np.random.rand(8, 8, 8, 8), 4)), kdims=['a', 'b', 'c', 'd'], vdims=['Value'])
    grouped = dataset.groupby(['c', 'd'])
    keys = list(product(range(8), range(8)))
    self.assertEqual(list(grouped.keys()), keys)
    for c, d in keys:
        self.assertEqual(grouped[c, d], dataset.select(c=c, d=d).reindex(['a', 'b']))