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
def test_deephash_set_inequality(self):
    self.assertNotEqual(deephash({1, 2, 3}), deephash({1, 3, 4}))