import datetime
import numpy as np
import pandas as pd
from holoviews import Dataset, Dimension, HoloMap
from holoviews.core.data import concat
from holoviews.core.data.interface import DataError
from holoviews.element import Curve, Scatter
from holoviews.element.comparison import ComparisonTestCase
from holoviews.util.transform import dim
def test_dataset_sample_ht(self):
    samples = self.dataset_ht.sample([0, 5, 10]).dimension_values('y')
    self.assertEqual(samples, np.array([0, 0.5, 1]))