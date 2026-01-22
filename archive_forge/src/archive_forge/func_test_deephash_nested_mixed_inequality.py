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
def test_deephash_nested_mixed_inequality(self):
    obj1 = [datetime.datetime(1, 2, 3), {1, 2, 3}, pd.DataFrame({'a': [1, 2], 'b': [3, 4]}), np.array([1, 2, 3]), {'a': 'b', '2': True}, dict([(1, 'a'), (2, 'b')]), np.int64(34)]
    obj2 = [datetime.datetime(1, 2, 3), {1, 2, 3}, pd.DataFrame({'a': [1, 2], 'b': [3, 4]}), np.array([1, 2, 3]), {'a': 'b', '1': True}, dict([(1, 'a'), (2, 'b')]), np.int64(34)]
    self.assertNotEqual(deephash(obj1), deephash(obj2))