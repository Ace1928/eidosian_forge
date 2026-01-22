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
def test_reindex_2d_grid_to_1d(self):
    with DatatypeContext([self.datatype, 'dictionary', 'dataframe'], self.dataset_grid):
        ds = self.dataset_grid.reindex(kdims=['x'])
    with DatatypeContext([self.datatype, 'dictionary', 'dataframe'], Dataset):
        self.assertEqual(ds, Dataset(self.dataset_grid.columns(), 'x', 'z'))