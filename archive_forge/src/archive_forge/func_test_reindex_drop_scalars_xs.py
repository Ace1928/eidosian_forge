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
def test_reindex_drop_scalars_xs(self):
    reindexed = self.dataset_grid.ndloc[:, 0].reindex()
    ds = Dataset((self.grid_ys, self.grid_zs[:, 0]), 'y', 'z')
    self.assertEqual(reindexed, ds)