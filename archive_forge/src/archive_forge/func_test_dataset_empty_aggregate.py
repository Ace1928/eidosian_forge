import datetime
import numpy as np
import pandas as pd
from holoviews import Dataset, Dimension, HoloMap
from holoviews.core.data import concat
from holoviews.core.data.interface import DataError
from holoviews.element import Curve, Scatter
from holoviews.element.comparison import ComparisonTestCase
from holoviews.util.transform import dim
def test_dataset_empty_aggregate(self):
    dataset = Dataset([], kdims=self.kdims, vdims=self.vdims)
    aggregated = Dataset([], kdims=self.kdims[:1], vdims=self.vdims)
    self.compare_dataset(dataset.aggregate(['Gender'], np.mean), aggregated)