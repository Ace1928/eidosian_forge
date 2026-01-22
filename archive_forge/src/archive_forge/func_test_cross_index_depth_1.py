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
def test_cross_index_depth_1(self):
    values = [self.values1]
    cross_product = list(product(*values))
    for i, p in enumerate(cross_product):
        self.assertEqual(cross_index(values, i), p)