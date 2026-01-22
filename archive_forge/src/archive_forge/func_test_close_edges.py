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
def test_close_edges(self):
    self.assertEqual(compute_edges(self.array2), np.array([0.25, 0.75, 1.25, 1.75]))