import datetime
import numpy as np
import pandas as pd
from holoviews import Dataset, Dimension, HoloMap
from holoviews.core.data import concat
from holoviews.core.data.interface import DataError
from holoviews.element import Curve, Scatter
from holoviews.element.comparison import ComparisonTestCase
from holoviews.util.transform import dim
def test_dataset_zip_init_alias(self):
    dataset = self.alias_table.clone(zip(self.gender, self.age, self.weight, self.height))
    self.assertTrue(isinstance(dataset.data, self.data_type))