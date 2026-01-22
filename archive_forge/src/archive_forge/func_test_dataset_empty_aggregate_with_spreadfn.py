import datetime
import numpy as np
import pandas as pd
from holoviews import Dataset, Dimension, HoloMap
from holoviews.core.data import concat
from holoviews.core.data.interface import DataError
from holoviews.element import Curve, Scatter
from holoviews.element.comparison import ComparisonTestCase
from holoviews.util.transform import dim
def test_dataset_empty_aggregate_with_spreadfn(self):
    dataset = Dataset([], kdims=self.kdims, vdims=self.vdims)
    aggregated = Dataset([], kdims=self.kdims[:1], vdims=[d for vd in self.vdims for d in [vd, vd + '_std']])
    self.compare_dataset(dataset.aggregate(['Gender'], np.mean, np.std), aggregated)