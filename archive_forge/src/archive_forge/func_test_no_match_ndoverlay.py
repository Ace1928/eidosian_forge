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
def test_no_match_ndoverlay(self):
    specs = [(0, ('Points', 'Points', '', 0)), (1, ('Points', 'Points', '', 1)), (2, ('Points', 'Points', '', 2))]
    spec = ('Scatter', 'Points', '', 5)
    self.assertEqual(closest_match(spec, specs), None)
    spec = ('Scatter', 'Bar', 'Foo', 5)
    self.assertEqual(closest_match(spec, specs), None)
    spec = ('Scatter', 'Foo', 'Bar', 5)
    self.assertEqual(closest_match(spec, specs), None)