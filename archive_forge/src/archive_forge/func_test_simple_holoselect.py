import datetime as dt
from itertools import product
import numpy as np
import pandas as pd
from holoviews.core import HoloMap
from holoviews.element import Contours, Curve, Image
from holoviews.element.comparison import ComparisonTestCase
def test_simple_holoselect(self):
    self.assertEqual(self.img_map.select(a=0, b=1), self.img_map[0, 1])