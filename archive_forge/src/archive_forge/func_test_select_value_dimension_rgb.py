import datetime as dt
from unittest import SkipTest
import numpy as np
from holoviews import HSV, RGB, Curve, Dataset, Dimension, Image, Table
from holoviews.core.data.interface import DataError
from holoviews.core.util import date_range
from .base import DatatypeContext, GriddedInterfaceTests, InterfaceTests
def test_select_value_dimension_rgb(self):
    self.assertEqual(self.rgb[..., 'R'], Image(np.flipud(self.rgb_array[:, :, 0]), bounds=self.rgb.bounds, vdims=[Dimension('R', range=(0, 1))], datatype=['image']))