import datetime as dt
from itertools import product
import numpy as np
import pandas as pd
from holoviews.core import HoloMap
from holoviews.element import Contours, Curve, Image
from holoviews.element.comparison import ComparisonTestCase
def test_deep_layout_nesting_slice(self):
    hmap1 = self.layout_map.HoloMap.I[1:3, 1:3, 0:0.5, 0:0.5]
    hmap2 = self.layout_map.HoloMap.II[1:3, 1:3, 0:0.5, 0:0.5]
    selection = self.layout_map.select(a=(1, 3), b=(1, 3), x=(0, 0.5), y=(0, 0.5))
    self.assertEqual(selection, hmap1 + hmap2)