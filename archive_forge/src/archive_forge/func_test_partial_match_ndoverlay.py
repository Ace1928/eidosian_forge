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
def test_partial_match_ndoverlay(self):
    spec = ('Points', 'Points', '', 5)
    specs = [(0, ('Points', 'Points', '', 0)), (1, ('Points', 'Points', '', 1)), (2, ('Points', 'Points', '', 2))]
    self.assertEqual(closest_match(spec, specs), 2)
    spec = ('Points', 'Points', 'Bar', 5)
    self.assertEqual(closest_match(spec, specs), 0)
    spec = ('Points', 'Foo', 'Bar', 5)
    self.assertEqual(closest_match(spec, specs), 0)