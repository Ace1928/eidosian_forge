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
def test_datetime64_to_s_int(self):
    dt = np.datetime64(datetime.datetime(2017, 1, 1))
    self.assertEqual(dt_to_int(dt, 's'), 1483228800.0)