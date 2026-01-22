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
def test_isfinite_pandas_timestamp_nat(self):
    dt64 = pd.Timestamp('NaT')
    self.assertFalse(isfinite(dt64))