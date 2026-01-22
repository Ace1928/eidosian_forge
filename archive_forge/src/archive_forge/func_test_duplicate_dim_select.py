import datetime as dt
from itertools import product
import numpy as np
import pandas as pd
from holoviews.core import HoloMap
from holoviews.element import Contours, Curve, Image
from holoviews.element.comparison import ComparisonTestCase
def test_duplicate_dim_select(self):
    selection = self.duplicate_map.select(x=(None, 0.5), y=(None, 0.5))
    self.assertEqual(selection, self.duplicate_map[:0.5, :0.5, :0.5, :0.5])