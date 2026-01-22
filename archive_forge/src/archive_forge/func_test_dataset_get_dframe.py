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
def test_dataset_get_dframe(self):
    df = self.dataset_hm.dframe()
    self.assertEqual(df.x.values, self.xs)
    self.assertEqual(df.y.values, self.y_ints.compute())