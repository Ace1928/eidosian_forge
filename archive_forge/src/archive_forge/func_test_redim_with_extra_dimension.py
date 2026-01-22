import datetime
import numpy as np
import pandas as pd
from holoviews import Dataset, Dimension, HoloMap
from holoviews.core.data import concat
from holoviews.core.data.interface import DataError
from holoviews.element import Curve, Scatter
from holoviews.element.comparison import ComparisonTestCase
from holoviews.util.transform import dim
def test_redim_with_extra_dimension(self):
    dataset = self.dataset_ht.add_dimension('Temp', 0, 0).clone(kdims=['x', 'y'], vdims=[])
    redimmed = dataset.redim(x='Time')
    self.assertEqual(redimmed.dimension_values('Time'), self.dataset_ht.dimension_values('x'))