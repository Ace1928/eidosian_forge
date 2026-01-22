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
def test_cross_index_large(self):
    values = [[chr(65 + i) for i in range(26)], list(range(500)), [chr(97 + i) for i in range(26)], [chr(48 + i) for i in range(10)]]
    self.assertEqual(cross_index(values, 50001), ('A', 192, 'i', '1'))
    self.assertEqual(cross_index(values, 500001), ('D', 423, 'c', '1'))