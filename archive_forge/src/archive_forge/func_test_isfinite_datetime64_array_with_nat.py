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
def test_isfinite_datetime64_array_with_nat(self):
    dts = [np.datetime64(datetime.datetime(2017, 1, i)) for i in range(1, 4)]
    dt64 = np.array(dts + [np.datetime64('NaT')])
    self.assertEqual(isfinite(dt64), np.array([True, True, True, False]))