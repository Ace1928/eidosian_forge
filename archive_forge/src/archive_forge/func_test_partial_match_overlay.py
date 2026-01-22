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
def test_partial_match_overlay(self):
    specs = [(0, ('Curve', 'Curve', 'I')), (1, ('Points', 'Points', 'I'))]
    spec = ('Curve', 'Curve')
    self.assertEqual(closest_match(spec, specs), 0)
    spec = ('Points', 'Points')
    self.assertEqual(closest_match(spec, specs), 1)