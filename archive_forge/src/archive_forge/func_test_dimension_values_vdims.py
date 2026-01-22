import datetime as dt
from unittest import SkipTest
import numpy as np
from holoviews import HSV, RGB, Curve, Dataset, Dimension, Image, Table
from holoviews.core.data.interface import DataError
from holoviews.core.util import date_range
from .base import DatatypeContext, GriddedInterfaceTests, InterfaceTests
def test_dimension_values_vdims(self):
    self.assertEqual(self.rgb.dimension_values(2, flat=False), self.rgb_array[:, :, 0])
    self.assertEqual(self.rgb.dimension_values(3, flat=False), self.rgb_array[:, :, 1])
    self.assertEqual(self.rgb.dimension_values(4, flat=False), self.rgb_array[:, :, 2])