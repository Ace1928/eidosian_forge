import datetime
import numpy as np
import pandas as pd
from holoviews import Dataset, Dimension, HoloMap
from holoviews.core.data import concat
from holoviews.core.data.interface import DataError
from holoviews.element import Curve, Scatter
from holoviews.element.comparison import ComparisonTestCase
from holoviews.util.transform import dim
def test_dataset_closest(self):
    closest = self.dataset_hm.closest([0.51, 1, 9.9])
    self.assertEqual(closest, [1.0, 1.0, 10.0])