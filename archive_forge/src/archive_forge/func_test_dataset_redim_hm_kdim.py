import datetime
import numpy as np
import pandas as pd
from holoviews import Dataset, Dimension, HoloMap
from holoviews.core.data import concat
from holoviews.core.data.interface import DataError
from holoviews.element import Curve, Scatter
from holoviews.element.comparison import ComparisonTestCase
from holoviews.util.transform import dim
def test_dataset_redim_hm_kdim(self):
    redimmed = self.dataset_hm.redim(x='Time')
    self.assertEqual(redimmed.dimension_values('Time'), self.dataset_hm.dimension_values('x'))