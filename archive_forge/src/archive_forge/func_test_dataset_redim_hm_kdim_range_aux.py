import datetime
import numpy as np
import pandas as pd
from holoviews import Dataset, Dimension, HoloMap
from holoviews.core.data import concat
from holoviews.core.data.interface import DataError
from holoviews.element import Curve, Scatter
from holoviews.element.comparison import ComparisonTestCase
from holoviews.util.transform import dim
def test_dataset_redim_hm_kdim_range_aux(self):
    redimmed = self.dataset_hm.redim.range(x=(-100, 3))
    self.assertEqual(redimmed.kdims[0].range, (-100, 3))