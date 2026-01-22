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
def test_no_streams_two_kdims(self):
    result = wrap_tuple_streams((1, 2), [Dimension('x'), Dimension('y')], [])
    self.assertEqual(result, (1, 2))