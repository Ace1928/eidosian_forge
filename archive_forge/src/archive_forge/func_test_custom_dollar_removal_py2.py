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
def test_custom_dollar_removal_py2(self):
    sanitize_identifier.eliminations.extend(['dollar'])
    sanitized = sanitize_identifier('$E$')
    self.assertEqual(sanitized, 'E')
    sanitize_identifier.eliminations.remove('dollar')