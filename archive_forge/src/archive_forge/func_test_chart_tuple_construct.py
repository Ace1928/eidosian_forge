import numpy as np
import pandas as pd
import param
from holoviews import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element.path import BaseShape
def test_chart_tuple_construct(self):
    self.assertEqual(Curve((self.xs, self.sin)), self.curve)