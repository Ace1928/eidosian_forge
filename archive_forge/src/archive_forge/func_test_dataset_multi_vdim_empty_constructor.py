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
def test_dataset_multi_vdim_empty_constructor(self):
    ds = Dataset([], ['x', 'y'], ['z1', 'z2', 'z3'])
    assert all((ds.dimension_values(vd, flat=False).shape == (0, 0) for vd in ds.vdims))