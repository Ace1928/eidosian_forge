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
def test_isfinite_float_array_inf(self):
    array = np.array([1.2, 3.0, np.inf])
    self.assertEqual(isfinite(array), np.array([True, True, False]))