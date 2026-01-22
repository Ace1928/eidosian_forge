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
def test_unique_array_categorial():
    ser = pd.Series(np.random.choice(['a', 'b', 'c'], 100)).astype('category')
    res = unique_array([ser])
    assert sorted(res) == ['a', 'b', 'c']