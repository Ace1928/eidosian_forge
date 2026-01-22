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
def test_get_path_from_item_with_custom_group_and_matching_label(self):
    path = get_path((('Custom', 'Path'), Element('Test', label='Path')))
    self.assertEqual(path, ('Custom', 'Path'))