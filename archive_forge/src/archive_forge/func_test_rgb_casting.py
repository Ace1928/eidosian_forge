import numpy as np
import pandas as pd
import param
from holoviews import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element.path import BaseShape
def test_rgb_casting(self):
    rgb = RGB([], bounds=2)
    self.assertEqual(rgb, RGB(rgb))