import datetime
import math
import unittest
from itertools import product
import numpy as np
import pandas as pd
from holoviews import Dimension, Element
from holoviews.core.util import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.streams import PointerXY
def test_compute_s_density_10s_datetime(self):
    start = np.datetime64(datetime.datetime.today())
    end = start + np.timedelta64(10, 's')
    self.assertEqual(compute_density(start, end, 10, 's'), 1)