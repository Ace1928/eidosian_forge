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
def test_isfinite_datetime(self):
    dt = datetime.datetime(2017, 1, 1)
    self.assertTrue(isfinite(dt))