import datetime
import numpy as np
import pandas as pd
from holoviews import Dataset, Dimension, HoloMap
from holoviews.core.data import concat
from holoviews.core.data.interface import DataError
from holoviews.element import Curve, Scatter
from holoviews.element.comparison import ComparisonTestCase
from holoviews.util.transform import dim
def test_dataset_redim_hm_kdim_soft_range_aux(self):
    redimmed = self.dataset_hm.redim.soft_range(x=(-100, 30))
    self.assertEqual(redimmed.kdims[0].soft_range, (-100, 30))