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
def test_dataset_2D_gridded_shape(self):
    array = da.from_array(np.random.rand(12, 11), 3)
    dataset = Dataset({'x': self.xs, 'y': range(12), 'z': array}, kdims=['x', 'y'], vdims=['z'])
    self.assertEqual(dataset.interface.shape(dataset, gridded=True), (12, 11))