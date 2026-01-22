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
def test_deephash_datetime_inequality(self):
    dt1 = datetime.datetime(1, 2, 3)
    dt2 = datetime.datetime(1, 2, 5)
    self.assertNotEqual(deephash(dt1), deephash(dt2))