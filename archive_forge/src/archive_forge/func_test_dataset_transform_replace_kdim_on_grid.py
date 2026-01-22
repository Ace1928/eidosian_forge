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
def test_dataset_transform_replace_kdim_on_grid(self):
    transformed = self.dataset_grid.transform(x=dim('x') * 2)
    expected = self.element(([0, 2], self.grid_ys, self.grid_zs), ['x', 'y'], ['z'])
    self.assertEqual(transformed, expected)