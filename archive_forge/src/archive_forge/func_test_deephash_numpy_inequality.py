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
def test_deephash_numpy_inequality(self):
    arr1 = np.array([1, 2, 3])
    arr2 = np.array([1, 2, 4])
    self.assertNotEqual(deephash(arr1), deephash(arr2))