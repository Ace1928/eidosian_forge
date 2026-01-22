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
def test_deephash_series_index_name_inequality(self):
    self.assertNotEqual(deephash(pd.Series([1, 2, 3], index=pd.Series([0, 1, 2], name='Foo'))), deephash(pd.Series([1, 2, 3], index=pd.Series([0, 1, 2], name='Bar'))))